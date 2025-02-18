from __future__ import annotations
import concurrent.futures
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
import zipfile
import tempfile
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Dict,
)
import requests

from darwin.datatypes import PathLike, Slot, SourceFile
from darwin.doc_enum import DocEnum
from darwin.path_utils import construct_full_path
from darwin.utils import chunk
from darwin.utils.utils import is_image_extension_allowed_by_filename, SLOTS_GRID_MAP

if TYPE_CHECKING:
    from darwin.client import Client
    from darwin.dataset import RemoteDataset
    from darwin.dataset.identifier import DatasetIdentifier

from abc import ABC, abstractmethod


class ItemMergeMode(Enum):
    SLOTS = "slots"
    SERIES = "series"
    CHANNELS = "channels"


class ItemPayload:
    """
    Represents an item's payload.

    Parameters
    ----------
    dataset_item_id : int
        The id of the dataset this item belongs to.
    filename : str
        The filename of where this ``ItemPayload``'s data is.
    path : str
        The path to ``filename``.
    reasons : Optional[List[str]], default: None
        A per-slot reason to upload this ``ItemPayload``.

    Attributes
    ----------
    dataset_item_id : int
        The id of the dataset this item belongs to.
    filename : str
        The filename of where this ``ItemPayload``'s data is.
    path : str
        The path to ``filename``.
    """

    def __init__(
        self,
        *,
        dataset_item_id: int,
        filename: str,
        path: str,
        reasons: Optional[List[str]] = None,
        slots: List[Dict[str, str]],
    ):
        self.dataset_item_id = dataset_item_id
        self.filename = filename
        self.path = PurePosixPath(path).as_posix()
        self.slots = [
            Slot(
                type=slot["type"],
                source_files=[SourceFile(file_name=slot["file_name"])],
                name=slot["slot_name"],
                upload_id=slot["upload_id"] if "upload_id" in slot else None,
                reason=slot["reason"] if "reason" in slot else None,
            )
            for slot in slots
        ]

    @staticmethod
    def parse_v2(payload):
        return ItemPayload(
            dataset_item_id=payload.get("id", None),
            filename=payload["name"],
            path=payload["path"],
            reasons=[slot.get("reason", None) for slot in payload["slots"]],
            slots=payload["slots"],
        )

    @property
    def full_path(self) -> str:
        """The full ``Path`` (with filename inclduded) to the file."""
        return construct_full_path(self.path, self.filename)


class UploadStage(DocEnum):
    """
    The different stages of uploading a file.
    """

    REQUEST_SIGNATURE = 0, "First stage, when authentication is being performed."
    UPLOAD_TO_S3 = 1, "Second stage, when the file is being uploaded to S3."
    CONFIRM_UPLOAD_COMPLETE = (
        2,
        "Final stage, when we confirm the file was correctly uploaded.",
    )
    OTHER = 3, "If the stage of the upload process is unknown."


@dataclass
class UploadRequestError(Exception):
    """
    Error throw when uploading a file fails with an unrecoverable error.
    """

    #: The ``Path`` of the file being uploaded.
    file_path: Path

    #: The ``UploadStage`` when the  failure happened.
    stage: UploadStage

    #: The ``Exception`` that triggered this unrecoverable error.
    error: Optional[Exception] = None


class LocalFile:
    """
    Represents a file locally stored.

    Parameters
    ----------
    local_path : PathLike
        The ``Path`` of the file.
    kwargs : Any
        Data relative to this file. Can be anything.

    Attributes
    ----------
    local_path : PathLike
        The ``Path`` of the file.
    data : Dict[str, str]
        Dictionary with metadata relative to this file. It has the following format:

        .. code-block:: python

            {
                "filename": "a_filename",
                "path": "a path"
            }

        - ``data["filename"]`` will hold the value passed as ``filename`` from ``kwargs`` or default to ``self.local_path.name``
        - ``data["path"]`` will hold the value passed as ``path`` from ``kwargs`` or default to ``"/"``

    """

    def __init__(
        self,
        local_path: PathLike,
        **kwargs,
    ):
        self.local_path = Path(local_path)
        self.data = kwargs
        self._type_check(kwargs)

    def _type_check(self, args) -> None:
        self.data["filename"] = args.get("filename") or self.local_path.name
        self.data["path"] = args.get("path") or "/"

    def serialize(self):
        return {
            "files": [{"file_name": self.data["filename"], "slot_name": "0"}],
            "name": self.data["filename"],
        }

    def serialize_darwin_json_v2(self):
        optional_properties = ["tags", "fps", "as_frames", "extract_views"]
        slot = {"file_name": self.data["filename"], "slot_name": "0"}
        for optional_property in optional_properties:
            if optional_property in self.data:
                slot[optional_property] = self.data.get(optional_property)

        serialized = {
            "slots": [slot],
            "name": self.data["filename"],
            "path": self.data["path"],
        }
        if "tags" in self.data:
            serialized["tags"] = self.data["tags"]
        return serialized

    @property
    def full_path(self) -> str:
        """The full ``Path`` (with filename inclduded) to the item."""
        return construct_full_path(self.data["path"], self.data["filename"])


class MultiFileItem:
    def __init__(
        self, directory: Path, files: List[Path], merge_mode: ItemMergeMode, fps: int
    ):
        self.directory = directory
        self.name = directory.name
        self.files = [LocalFile(file, fps=fps) for file in files]
        self.merge_mode = merge_mode
        self._prepare_local_files_and_create_layout()

    def _prepare_local_files_and_create_layout(self):
        """
        This function:
        - Ensures that the files to be uploaded are valid for the given merge mode
        - Creates a LayoutV3 object for `ItemMergeMode.SLOTS` & `ItemMergeMode.CHANNELS` items

        For `ItemMergeMode.SERIES` items:
        - Every slice is zipped into a single source file. This is necessary to upload
        individual DICOM slices as volumetric series
        - Layout is set to `None` because the files are zipped into a single source file

        Raises
        ------
        ValueError
            - If no DICOM files are found in the directory for `ItemMergeMode.SERIES` items
            - If the number of files is greater than 16 for `ItemMergeMode.CHANNELS` items
        """
        self.slot_names = []
        if self.merge_mode == ItemMergeMode.SLOTS:
            num_viewports = min(len(self.files), 16)
            slots_grid = SLOTS_GRID_MAP[num_viewports]
            self.layout = {
                "version": 3,
                "slots_grid": slots_grid,
            }
            self.slot_names = [str(i) for i in range(len(self.files))]
        elif self.merge_mode == ItemMergeMode.SERIES:
            self.files = [
                file for file in self.files if file.local_path.suffix.lower() == ".dcm"
            ]
            if not self.files:
                raise ValueError("No `.dcm` files found in 1st level of directory")
            self._create_series_zip()
            self.layout = None
            self.slot_names = ["0"]
        elif self.merge_mode == ItemMergeMode.CHANNELS:
            # Currently, only image files are supported in multi-channel items. This is planned to change in the future
            self.files = [
                file
                for file in self.files
                if is_image_extension_allowed_by_filename(str(file.local_path))
            ]
            if not self.files:
                raise ValueError(
                    "No supported filetypes found in 1st level of directory. Currently, multi-channel items only support images."
                )
            if len(self.files) > 16:
                raise ValueError(
                    f"No multi-channel item can have more than 16 files. The following directory has {len(self.files)} files: {self.directory}"
                )
            self.layout = {
                "version": 3,
                "slots_grid": [[[file.local_path.name for file in self.files]]],
            }
            self.slot_names = self.layout["slots_grid"][0][0]

    def serialize_darwin_json_v2(self):
        optional_properties = ["fps"]
        slots = []
        for idx, local_file in enumerate(self.files):
            slot = {
                "file_name": local_file.data["filename"],
                "slot_name": self.slot_names[idx],
            }
            for optional_property in optional_properties:
                if optional_property in local_file.data:
                    slot[optional_property] = local_file.data.get(optional_property)
            slots.append(slot)

        return {
            "slots": slots,
            "layout": self.layout,
            "name": self.name,
            "path": "/",
        }

    def _create_series_zip(self):
        """
        For a given series `MultiFileItem`:
        - Zip all `.dcm` files into a temporary zip file
        - Replace all `.dcm` files with the zip file

        This is necessary to upload individual DICOM slices as volumetric series
        """
        self._temp_zip_dir = tempfile.TemporaryDirectory()
        zip_path = Path(self._temp_zip_dir.name) / f"{self.name}.dcm"
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for local_file in self.files:
                zip_file.write(local_file.local_path, local_file.local_path.name)
        self.files = [LocalFile(zip_path)]

    @property
    def full_path(self) -> str:
        """The full ``Path`` (with filename included) to the item"""
        return "/" + self.name


class FileMonitor(object):
    """
    Monitors the progress of a :class:``BufferedReader``.

    To use this monitor, you construct your :class:``BufferedReader`` as you
    normally would, then construct this object with it as argument.

    Parameters
    ----------
    io : BinaryIO
        IO object used by this class. Depency injection.
    file_size : int
        The fie of the file in bytes.
    callback : Callable[["FileMonitor"], None]
        Callable function used by this class. Depency injection via constructor.

    Attributes
    ----------
    io : BinaryIO
        IO object used by this class. Depency injection.
    callback : Callable[["FileMonitor"], None]
        Callable function used by this class. Depency injection.
    bytes_read : int
      Amount of bytes read from the IO.
    len : int
        Total size of the IO.
    """

    def __init__(
        self, io: BinaryIO, file_size: int, callback: Callable[["FileMonitor"], None]
    ):
        self.io: BinaryIO = io
        self.callback: Callable[["FileMonitor"], None] = callback

        self.bytes_read: int = 0
        self.len: int = file_size

    def read(self, size: int = -1) -> Any:
        """
        Reads given amount of bytes from configured IO and calls the configured callback for each
        block read. The callback is passed a reference this object that can be used to get current
        self.bytes_read.

        Parameters
        ----------
        size : int, default: -1
            The number of bytes to read. Defaults to -1, so all bytes until EOF are read.

        Returns
        -------
        data: Any
            Data read from the IO.
        """
        data: Any = self.io.read(size)
        self.bytes_read += len(data)
        self.callback(self)

        return data


ByteReadCallback = Callable[[Optional[str], float, float], None]
ProgressCallback = Callable[[int, float], None]
FileUploadCallback = Callable[[str, int, int], None]


class UploadHandler(ABC):
    """
    Holds responsibilities for file upload management and failure into ``RemoteDataset``\\s.

    Parameters
    ----------
    dataset: RemoteDataset
        Target ``RemoteDataset`` where we want to upload our files to.
    uploading_files : Union[List[LocalFile], List[MultiFileItems]]
        List of ``LocalFile``\\s or ``MultiFileItem``\\s to be uploaded.

    Attributes
    ----------
    dataset : RemoteDataset
        Target ``RemoteDataset`` where we want to upload our files to.
    errors : List[UploadRequestError]
        List of errors that happened during the upload process
    local_files : List[LocalFile]
        List of ``LocalFile``\\s to be uploaded.
    multi_file_items : List[MultiFileItem]
        List of ``MultiFileItem``\\s to be uploaded.
    blocked_items : List[ItemPayload]
        List of items that were not able to be uploaded.
    pending_items : List[ItemPayload]
        List of items waiting to be uploaded.
    """

    def __init__(
        self,
        dataset: "RemoteDataset",
        local_files: List[LocalFile],
        multi_file_items: Optional[List[MultiFileItem]] = None,
        handle_as_slices: Optional[bool] = False,
    ):
        self._progress: Optional[
            Iterator[Callable[[Optional[ByteReadCallback]], None]]
        ] = None
        self.multi_file_items = multi_file_items
        self.local_files = local_files
        self.dataset: RemoteDataset = dataset
        self.errors: List[UploadRequestError] = []
        self.blocked_items, self.pending_items = self._request_upload(
            handle_as_slices=handle_as_slices
        )

    @staticmethod
    def build(
        dataset: "RemoteDataset",
        local_files: List[LocalFile],
        handle_as_slices: Optional[bool] = False,
    ):
        return UploadHandlerV2(dataset, local_files, handle_as_slices=handle_as_slices)

    @property
    def client(self) -> "Client":
        """The ``Client`` used by this ``UploadHander``\\'s ``RemoteDataset``."""
        return self.dataset.client

    @property
    def dataset_identifier(self) -> "DatasetIdentifier":
        """The ``DatasetIdentifier`` of this ``UploadHander``\\'s ``RemoteDataset``."""
        return self.dataset.identifier

    @property
    def blocked_count(self) -> int:
        """Number of items that could not be uploaded successfully."""
        return len(self.blocked_items)

    @property
    def error_count(self) -> int:
        """Number of errors that prevented items from being uploaded."""
        return len(self.errors)

    @property
    def pending_count(self) -> int:
        """Number of items waiting to be uploaded."""
        return len(self.pending_items)

    @property
    def total_count(self) -> int:
        """Total number of blocked and pending items."""
        return self.pending_count + self.blocked_count

    @property
    def progress(self):
        """Current level of upload progress."""
        return self._progress

    def prepare_upload(
        self,
    ) -> Optional[Iterator[Callable[[Optional[ByteReadCallback]], None]]]:
        self._progress = self._upload_files()
        return self._progress

    def upload(
        self,
        multi_threaded: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
        file_upload_callback: Optional[FileUploadCallback] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        if not self._progress:
            self.prepare_upload()

        if progress_callback:
            progress_callback(self.pending_count, 0)

        # needed to ensure that we don't mark a file as completed twice
        file_complete: Set[str] = set()

        def callback(file_name, file_total_bytes, file_bytes_sent):
            if file_upload_callback:
                file_upload_callback(file_name, file_total_bytes, file_bytes_sent)

            if progress_callback:
                if (
                    file_total_bytes == file_bytes_sent
                    and file_name not in file_complete
                ):
                    file_complete.add(file_name)
                    progress_callback(self.pending_count, 1)

        if max_workers:
            if max_workers < 1:
                raise ValueError("max_workers must be greater than 0")
            elif max_workers > concurrent.futures.ThreadPoolExecutor()._max_workers:
                raise ValueError(
                    f"max_workers must be less than or equal to {concurrent.futures.ThreadPoolExecutor()._max_workers}"
                )

        if multi_threaded and self.progress:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_progress = {
                    executor.submit(f, callback) for f in self.progress
                }
                for future in concurrent.futures.as_completed(future_to_progress):
                    try:
                        future.result()
                    except Exception as exc:
                        print("exception", exc)
        elif self.progress:
            for file_to_upload in self.progress:
                file_to_upload(callback)

    @abstractmethod
    def _request_upload(
        self, handle_as_slices: Optional[bool] = False
    ) -> Tuple[List[ItemPayload], List[ItemPayload]]:
        pass

    @abstractmethod
    def _upload_files(self) -> Iterator[Callable[[Optional[ByteReadCallback]], None]]:
        pass

    @abstractmethod
    def _upload_file(
        self,
        dataset_item_id: int,
        file_path: Path,
        byte_read_callback: Optional[ByteReadCallback],
    ) -> None:
        pass


class UploadHandlerV2(UploadHandler):
    def __init__(
        self,
        dataset: "RemoteDataset",
        local_files: List[LocalFile],
        multi_file_items: Optional[List[MultiFileItem]] = None,
        handle_as_slices: Optional[bool] = False,
    ):
        super().__init__(
            dataset=dataset,
            local_files=local_files,
            multi_file_items=multi_file_items,
            handle_as_slices=handle_as_slices,
        )

    def _request_upload(
        self, handle_as_slices: Optional[bool] = False
    ) -> Tuple[List[ItemPayload], List[ItemPayload]]:
        blocked_items = []
        items = []
        chunk_size: int = _upload_chunk_size()
        single_file_items = self.local_files
        upload_payloads = []
        if self.multi_file_items:
            upload_payloads.extend(
                [
                    {
                        "items": [
                            file.serialize_darwin_json_v2() for file in file_chunk
                        ],
                        "options": {
                            "handle_as_slices": handle_as_slices,
                        },
                    }
                    for file_chunk in chunk(self.multi_file_items, chunk_size)
                ]
            )
            local_files_for_multi_file_items = [
                file
                for multi_file_item in self.multi_file_items
                for file in multi_file_item.files
            ]
            single_file_items = [
                file
                for file in single_file_items
                if file not in local_files_for_multi_file_items
            ]

        upload_payloads.extend(
            [
                {
                    "items": [file.serialize_darwin_json_v2() for file in file_chunk],
                    "options": {"handle_as_slices": handle_as_slices},
                }
                for file_chunk in chunk(single_file_items, chunk_size)
            ]
        )

        dataset_slug: str = self.dataset_identifier.dataset_slug
        team_slug: Optional[str] = self.dataset_identifier.team_slug
        for upload_payload in upload_payloads:
            data: Dict[str, Any] = self.client.api_v2.register_data(
                dataset_slug, upload_payload, team_slug=team_slug
            )
            blocked_items.extend(
                [ItemPayload.parse_v2(item) for item in data["blocked_items"]]
            )
            items.extend([ItemPayload.parse_v2(item) for item in data["items"]])
        return blocked_items, items

    def _upload_files(self) -> Iterator[Callable[[Optional[ByteReadCallback]], None]]:
        def upload_function(
            dataset_slug, local_path, upload_id
        ) -> Callable[[Optional[ByteReadCallback]], None]:
            return lambda byte_read_callback=None: self._upload_file(
                dataset_slug, local_path, upload_id, byte_read_callback
            )

        file_lookup = {file.full_path: file for file in self.local_files}
        for item in self.pending_items:
            for slot in item.slots:
                upload_id = slot.upload_id
                slot_path = (
                    Path(item.path) / Path((slot.source_files[0].file_name))
                ).as_posix()
                file = file_lookup.get(str(slot_path))
                if not file:
                    raise ValueError(
                        f"Cannot match {slot_path} from payload with files to upload"
                    )
                yield upload_function(
                    self.dataset.identifier.dataset_slug, file.local_path, upload_id
                )

    def _upload_file(
        self,
        dataset_slug: str,
        file_path: Path,
        upload_id: str,
        byte_read_callback: Optional[ByteReadCallback],
    ) -> None:
        try:
            self._do_upload_file(dataset_slug, file_path, upload_id, byte_read_callback)
        except UploadRequestError as e:
            self.errors.append(e)
        except Exception as e:
            self.errors.append(
                UploadRequestError(
                    file_path=file_path, stage=UploadStage.OTHER, error=e
                )
            )

    def _do_upload_file(
        self,
        dataset_slug: str,
        file_path: Path,
        upload_id: str,
        byte_read_callback: Optional[ByteReadCallback] = None,
    ) -> None:
        team_slug: Optional[str] = self.dataset_identifier.team_slug

        try:
            sign_response: Dict[str, Any] = self.client.api_v2.sign_upload(
                dataset_slug, upload_id, team_slug=team_slug
            )
        except Exception as e:
            raise UploadRequestError(
                file_path=file_path, stage=UploadStage.REQUEST_SIGNATURE, error=e
            )

        upload_url = sign_response["upload_url"]

        try:
            file_size = file_path.stat().st_size
            if byte_read_callback:
                byte_read_callback(str(file_path), file_size, 0)

            def callback(monitor):
                if byte_read_callback:
                    byte_read_callback(str(file_path), file_size, monitor.bytes_read)

            with file_path.open("rb") as m:
                monitor = FileMonitor(m, file_size, callback)

                retries = 0
                while retries < 5:
                    upload_response = requests.put(f"{upload_url}", data=monitor)
                    # If s3 is getting to many request it will return 503, we will sleep and retry
                    if upload_response.status_code != 503:
                        break

                    time.sleep(2**retries)
                    retries += 1

            upload_response.raise_for_status()
        except Exception as e:
            raise UploadRequestError(
                file_path=file_path, stage=UploadStage.UPLOAD_TO_S3, error=e
            )

        try:
            self.client.api_v2.confirm_upload(
                dataset_slug, upload_id, team_slug=team_slug
            )
        except Exception as e:
            raise UploadRequestError(
                file_path=file_path, stage=UploadStage.CONFIRM_UPLOAD_COMPLETE, error=e
            )


DEFAULT_UPLOAD_CHUNK_SIZE: int = 500


def _upload_chunk_size() -> int:
    """
    Gets the chunk size to be used from the OS environment, or uses the default one if that is not
    possible. The default chunk size is 500.

    Returns
    -------
    int
        The chunk size to be used.
    """
    env_chunk: Optional[str] = os.getenv("DARWIN_UPLOAD_CHUNK_SIZE")
    if env_chunk is None:
        return DEFAULT_UPLOAD_CHUNK_SIZE

    try:
        return int(env_chunk)
    except ValueError:
        print("Cannot cast environment variable DEFAULT_UPLOAD_CHUNK_SIZE to integer")
        print(f"Setting chunk size to {DEFAULT_UPLOAD_CHUNK_SIZE}")
        return DEFAULT_UPLOAD_CHUNK_SIZE
