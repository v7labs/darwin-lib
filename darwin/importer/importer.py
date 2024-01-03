from collections import defaultdict
from logging import getLogger
from multiprocessing import cpu_count
from pathlib import Path
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from darwin.datatypes import AnnotationFile, Property, parse_property_classes
from darwin.future.data_objects.properties import (
    FullProperty,
    PropertyType,
    PropertyValue,
    SelectedProperty,
)
from darwin.item import DatasetItem
from darwin.path_utils import is_properties_enabled, parse_metadata

Unknown = Any  # type: ignore

from tqdm import tqdm

if TYPE_CHECKING:
    from darwin.client import Client
    from darwin.dataset.remote_dataset import RemoteDataset

import deprecation
from rich.console import Console
from rich.progress import track
from rich.theme import Theme

import darwin.datatypes as dt
from darwin.datatypes import PathLike
from darwin.exceptions import IncompatibleOptions, RequestEntitySizeExceeded
from darwin.utils import secure_continue_request
from darwin.utils.flatten_list import flatten_list
from darwin.version import __version__

logger = getLogger(__name__)

try:
    from mpire import WorkerPool

    MPIRE_AVAILABLE = True
except ImportError:
    MPIRE_AVAILABLE = False

# Classes missing import support on backend side
UNSUPPORTED_CLASSES = ["string", "graph"]

# Classes that are defined on team level automatically and available in all datasets
GLOBAL_CLASSES = ["__raster_layer__"]

DEPRECATION_MESSAGE = """

This function is going to be turned into private. This means that breaking
changes in its interface and implementation are to be expected. We encourage using ``import_annotations``
instead of calling this low-level function directly.

"""


@deprecation.deprecated(  # type:ignore
    deprecated_in="0.7.12",
    removed_in="0.8.0",
    current_version=__version__,
    details=DEPRECATION_MESSAGE,
)
def build_main_annotations_lookup_table(
    annotation_classes: List[Dict[str, Unknown]]
) -> Dict[str, Unknown]:
    MAIN_ANNOTATION_TYPES = [
        "bounding_box",
        "cuboid",
        "ellipse",
        "keypoint",
        "line",
        "link",
        "polygon",
        "skeleton",
        "tag",
        "string",
        "table",
        "graph",
        "mask",
        "raster_layer",
    ]
    lookup: Dict[str, Unknown] = {}
    for cls in annotation_classes:
        for annotation_type in cls["annotation_types"]:
            if annotation_type in MAIN_ANNOTATION_TYPES:
                if annotation_type not in lookup:
                    lookup[annotation_type] = {}
                lookup[annotation_type][cls["name"]] = cls["id"]

    return lookup


@deprecation.deprecated(  # type:ignore
    deprecated_in="0.7.12",
    removed_in="0.8.0",
    current_version=__version__,
    details=DEPRECATION_MESSAGE,
)
def find_and_parse(  # noqa: C901
    importer: Callable[[Path], Union[List[dt.AnnotationFile], dt.AnnotationFile, None]],
    file_paths: List[PathLike],
    console: Optional[Console] = None,
    use_multi_cpu: bool = True,
    cpu_limit: int = 1,
) -> Optional[Iterable[dt.AnnotationFile]]:
    is_console = console is not None

    logger = getLogger(__name__)

    def perf_time(reset: bool = False) -> Generator[float, float, None]:
        start = perf_counter()
        yield start
        while True:
            if reset:
                start = perf_counter()
            yield perf_counter() - start

    def maybe_console(*args: Union[str, int, float]) -> None:
        if console is not None:
            console.print(*[f"[{str(next(perf_time()))} seconds elapsed]", *args])
        else:
            logger.info(*[f"[{str(next(perf_time()))}]", *args])

    maybe_console("Parsing files... ")

    files: List[Path] = _get_files_for_parsing(file_paths)

    maybe_console(f"Found {len(files)} files")

    if use_multi_cpu and MPIRE_AVAILABLE and cpu_limit > 1:
        maybe_console(f"Using multiprocessing with {cpu_limit} workers")
        try:
            with WorkerPool(cpu_limit) as pool:
                parsed_files = pool.map(importer, files, progress_bar=is_console)
        except KeyboardInterrupt:
            maybe_console("Keyboard interrupt. Stopping.")
            return None
        except Exception as e:
            maybe_console(f"Error: {e}")
            return None

    else:
        maybe_console("Using single CPU")
        parsed_files = list(map(importer, tqdm(files) if is_console else files))

    maybe_console("Finished.")
    # Sometimes we have a list of lists of AnnotationFile, sometimes we have a list of AnnotationFile
    # We flatten the list of lists
    if isinstance(parsed_files, list):
        if isinstance(parsed_files[0], list):
            parsed_files = [item for sublist in parsed_files for item in sublist]
    else:
        parsed_files = [parsed_files]

    parsed_files = [f for f in parsed_files if f is not None]
    return parsed_files


def _get_files_for_parsing(file_paths: List[PathLike]) -> List[Path]:
    packed_files = [
        filepath.glob("**/*") if filepath.is_dir() else [filepath]
        for filepath in map(Path, file_paths)
    ]
    return [file for files in packed_files for file in files]


@deprecation.deprecated(  # type:ignore
    deprecated_in="0.7.12",
    removed_in="0.8.0",
    current_version=__version__,
    details=DEPRECATION_MESSAGE,
)
def build_attribute_lookup(dataset: "RemoteDataset") -> Dict[str, Unknown]:
    attributes: List[Dict[str, Unknown]] = dataset.fetch_remote_attributes()
    lookup: Dict[str, Unknown] = {}
    for attribute in attributes:
        class_id = attribute["class_id"]
        if class_id not in lookup:
            lookup[class_id] = {}
        lookup[class_id][attribute["name"]] = attribute["id"]
    return lookup


@deprecation.deprecated(  # type:ignore
    deprecated_in="0.7.12",
    removed_in="0.8.0",
    current_version=__version__,
    details=DEPRECATION_MESSAGE,
)
def get_remote_files(
    dataset: "RemoteDataset", filenames: List[str], chunk_size: int = 100
) -> Dict[str, Tuple[int, str]]:
    """
    Fetches remote files from the datasets in chunks; by default 100 filenames at a time.

    The output is a two-element tuple of:
    - file ID
    - the name of the first slot for V2 items, or '0' for V1 items

    Fetching slot name is necessary here to avoid double-trip to Api downstream for remote files.
    """
    remote_files = {}
    for i in range(0, len(filenames), chunk_size):
        chunk = filenames[i : i + chunk_size]
        for remote_file in dataset.fetch_remote_files(
            {"types": "image,playback_video,video_frame", "filenames": chunk}
        ):
            slot_name = _get_slot_name(remote_file)
            remote_files[remote_file.full_path] = (remote_file.id, slot_name)
    return remote_files


def _get_slot_name(remote_file: DatasetItem) -> str:
    slot = next(iter(remote_file.slots), {"slot_name": "0"})
    if slot:
        return slot["slot_name"]
    else:
        return "0"


def _resolve_annotation_classes(
    local_annotation_classes: List[dt.AnnotationClass],
    classes_in_dataset: Dict[str, Unknown],
    classes_in_team: Dict[str, Unknown],
) -> Tuple[Set[dt.AnnotationClass], Set[dt.AnnotationClass]]:
    local_classes_not_in_dataset: Set[dt.AnnotationClass] = set()
    local_classes_not_in_team: Set[dt.AnnotationClass] = set()

    for local_cls in local_annotation_classes:
        local_annotation_type = (
            local_cls.annotation_internal_type or local_cls.annotation_type
        )
        # Only add the new class if it doesn't exist remotely already
        if (
            local_annotation_type in classes_in_dataset
            and local_cls.name in classes_in_dataset[local_annotation_type]
        ):
            continue

        # Only add the new class if it's not included in the list of the missing classes already
        if local_cls.name in [
            missing_class.name for missing_class in local_classes_not_in_dataset
        ]:
            continue
        if local_cls.name in [
            missing_class.name for missing_class in local_classes_not_in_team
        ]:
            continue

        if (
            local_annotation_type in classes_in_team
            and local_cls.name in classes_in_team[local_annotation_type]
        ):
            local_classes_not_in_dataset.add(local_cls)
        else:
            local_classes_not_in_team.add(local_cls)

    return local_classes_not_in_dataset, local_classes_not_in_team


def _update_payload_with_properties(
        annotations: List[Dict[str, Unknown]],
        annotation_id_property_map: Dict[int, Dict[str, List[str]]]
    ) -> None:
    """
    Updates the annotations with the properties that were created/updated during the import.
    """
    if not annotation_id_property_map:
        return

    for annotation in annotations:
        annotation_id = annotation["annotation_class_id"]
        if annotation_id_property_map.get(annotation_id):
            annotation["annotation_properties"] = {"0": annotation_id_property_map[annotation_id]}

def _import_properties(
    metadata_path: Union[Path, bool],
    client: "Client",
    annotations: List[dt.Annotation],
    annotation_class_ids_map: Dict[str, str],
) -> Dict[int, Dict[str, List[str]]]:
    annotation_id_property_map: Dict[int, Dict[str, List[str]]] = {}
    if not isinstance(metadata_path, Path):
        # No properties to import
        return

    # parse metadata.json file -> list[PropertyClass]
    metadata = parse_metadata(metadata_path)
    metadata_property_classes = parse_property_classes(metadata)

    # get team properties -> List[FullProperty]
    team_properties = client.get_team_properties()
    # (property-name, annotation_class_id): FullProperty object
    team_properties_annotation_lookup: Dict[
        Tuple[str, Optional[int]], FullProperty
    ] = {}
    for prop in team_properties:
        team_properties_annotation_lookup[(prop.name, prop.annotation_class_id)] = prop

    # (annotation-cls-name, annotation-cls-name): PropertyClass object
    metadata_classes_lookup: List[Tuple[str, str]] = []
    # (annotation-cls-name, property-name): Property object
    metadata_cls_prop_lookup: Dict[Tuple[str, str], Property] = {}
    for _cls in metadata_property_classes:
        metadata_classes_lookup += [(_cls.name, _cls.type)]
        for _prop in _cls.properties or []:
            metadata_cls_prop_lookup[(_cls.name, _prop.name)] = _prop

    create_properties: List[FullProperty] = []
    update_properties: List[FullProperty] = []
    for annotation in annotations:
        annotation_name = annotation.annotation_class.name
        annotation_type = annotation_type = (
            annotation.annotation_class.annotation_internal_type
            or annotation.annotation_class.annotation_type
        )
        annotation_class_id = int(annotation_class_ids_map[annotation_name])
        annotation_id_property_map[annotation_class_id] = defaultdict(list)

        # raise error if annotation class not present in metadata
        if (annotation_name, annotation_type) not in metadata_classes_lookup:
            _msg = f"Annotation: '{annotation_name}' not found in {metadata_path}"
            raise ValueError(_msg)

        # loop on annotation properties and check if they exist in metadata & team
        for a_prop in annotation.properties or []:
            a_prop: SelectedProperty

            # raise error if annotation-property not present in metadata
            if (annotation_name, a_prop.name) not in metadata_cls_prop_lookup:
                _msg = f"Annotation: '{annotation_name}' -> Property '{a_prop.name}' not found in {metadata_path}"
                raise ValueError(_msg)

            # get metadata property
            m_prop: Property = metadata_cls_prop_lookup[(annotation_name, a_prop.name)]

            # get metadata property type
            m_prop_type: PropertyType = m_prop.type

            # get metadata property options
            m_prop_options: List[Dict[str, str]] = m_prop.options or []

            # check if property value is missing for a property that requires a value
            if m_prop.required and not a_prop.value:
                _msg = f"Annotation: '{annotation_name}' -> Property '{a_prop.name}' requires a value!"
                raise ValueError(_msg)

            # check if property and annotation class exists in team
            if (a_prop.name, annotation_class_id) \
                not in team_properties_annotation_lookup:

                # if it doesn't exist, create it
                full_property = FullProperty(
                    name=a_prop.name,
                    type=m_prop_type,  # type from .v7/metadata.json
                    required=m_prop.required,  # required from .v7/metadata.json
                    description="property-created-during-annotation-import",
                    slug=client.default_team,
                    annotation_class_id=int(annotation_class_id),
                    property_values=[
                        PropertyValue(
                            position=m_prop_option.get("position"),  # type: ignore
                            type=m_prop_option.get("type"),  # type: ignore
                            value=m_prop_option.get("value"),  # type: ignore
                            color=m_prop_option.get("color"),  # type: ignore
                        )
                        for m_prop_option in m_prop_options  # options from .v7/metadata.json
                    ],
                )
                create_properties.append(full_property)
                continue

            # check if property value/type is different in m_prop (.v7/metadata.json) options
            for m_prop_option in m_prop_options:
                if m_prop_option.get("value") == a_prop.value and m_prop_option.get("type") == a_prop.type:
                    break
            else:
                _msg = f"Annotation: '{annotation_name}' -> Property '{a_prop.value}' ({a_prop.type}) not found in .v7/metadata.json, found: {m_prop.options}"
                raise ValueError(_msg)

            # get team property
            t_prop: FullProperty = team_properties_annotation_lookup[
                (a_prop.name, annotation_class_id)
            ]

            # check if property value is missing for a property that requires a value
            if t_prop.required and not a_prop.value:
                _msg = f"Annotation: '{annotation_name}' -> Property '{a_prop.name}' requires a value!"
                raise ValueError(_msg)

            # check if property value/type is different in t_prop (team) options
            for t_prop_val in t_prop.property_values or []:
                if t_prop_val.value == {"value": a_prop.value} and t_prop_val.type == a_prop.type:
                    break
            else:
                # if it is, update it
                full_property = FullProperty(
                    id=t_prop.id,
                    name=a_prop.name,
                    type=m_prop_type,
                    required=m_prop.required,
                    description="property-updated-during-annotation-import",
                    slug=client.default_team,
                    annotation_class_id=int(annotation_class_id),
                    property_values=[
                        PropertyValue(
                            type=m_prop_option.get("type"),  # type: ignore
                            value={"value": m_prop_option.get("value")},  # type: ignore
                            color=m_prop_option.get("color"),  # type: ignore
                        )
                    ],
                )
                update_properties.append(full_property)
                continue

            assert t_prop.id is not None
            assert t_prop_val.id is not None
            annotation_id_property_map[annotation_class_id][t_prop.id].append(t_prop_val.id)

    console = Console(theme=_console_theme())
    if create_properties:
        console.print(f"Creating {len(create_properties)} properties", style="info")
        for full_property in create_properties:
            _msg = f"Creating property {full_property.name} ({full_property.type})"
            console.print(_msg, style="info")
            prop = client.create_property(team_slug=full_property.slug, params=full_property)

            assert full_property.annotation_class_id is not None
            assert prop.id is not None
            annotation_id_property_map[full_property.annotation_class_id][prop.id] = []
            for prop_val in prop.property_values or []:
                assert prop_val.id is not None
                annotation_id_property_map[full_property.annotation_class_id][prop.id].append(
                    prop_val.id
                )

    if update_properties:
        console.print(f"Updating {len(update_properties)} properties", style="info")
        for full_property in update_properties:
            _msg = f"Updating property {full_property.name} ({full_property.type})"
            console.print(_msg, style="info")
            client.update_property(team_slug=full_property.slug, params=full_property)

            assert full_property.annotation_class_id is not None
            assert full_property.id is not None
            annotation_id_property_map[full_property.annotation_class_id][full_property.id] = []
            for prop_val in full_property.property_values or []:
                assert prop_val.id is not None
                annotation_id_property_map[full_property.annotation_class_id][full_property.id].append(
                    prop_val.id
                )

    return annotation_id_property_map


def import_annotations(  # noqa: C901
    dataset: "RemoteDataset",
    importer: Callable[[Path], Union[List[dt.AnnotationFile], dt.AnnotationFile, None]],
    file_paths: List[PathLike],
    append: bool,
    class_prompt: bool = True,
    delete_for_empty: bool = False,
    import_annotators: bool = False,
    import_reviewers: bool = False,
    use_multi_cpu: bool = False,  # Set to False to give time to resolve MP behaviours
    cpu_limit: Optional[int] = None,  # 0 because it's set later in logic
) -> None:
    """
    Imports the given given Annotations into the given Dataset.

    Parameters
    ----------
    dataset : RemoteDataset
        Dataset where the Annotations will be imported to.
    importer : Callable[[Path], Union[List[dt.AnnotationFile], dt.AnnotationFile, None]]
        Parsing module containing the logic to parse the given Annotation files given in
        ``files_path``. See ``importer/format`` for a list of out of supported parsers.
    file_paths : List[PathLike]
        A list of ``Path``'s or strings containing the Annotations we wish to import.
    append : bool
        If ``True`` appends the given annotations to the datasets. If ``False`` will override them.
        Incompatible with ``delete-for-empty``.
    class_prompt : bool
        If ``False`` classes will be created and added to the datasets without requiring a user's prompt.
    delete_for_empty : bool, default: False
        If ``True`` will use empty annotation files to delete all annotations from the remote file.
        If ``False``, empty annotation files will simply be skipped.
        Only works for V2 datasets.
        Incompatible with ``append``.
    import_annotators : bool, default: False
        If ``True`` it will import the annotators from the files to the dataset, if available.
        If ``False`` it will not import the annotators.
    import_reviewers : bool, default: False
        If ``True`` it will import the reviewers from the files to the dataset, if .
        If ``False`` it will not import the reviewers.
    use_multi_cpu : bool, default: True
        If ``True`` will use multiple available CPU cores to parse the annotation files.
        If ``False`` will use only the current Python process, which runs in one core.
        Processing using multiple cores is faster, but may slow down a machine also running other processes.
        Processing with one core is slower, but will run well alongside other processes.
    cpu_limit : int, default: 2 less than total cpu count
        The maximum number of CPU cores to use when ``use_multi_cpu`` is ``True``.
        If ``cpu_limit`` is greater than the number of available CPU cores, it will be set to the number of available cores.
        If ``cpu_limit`` is less than 1, it will be set to CPU count - 2.
        If ``cpu_limit`` is omitted, it will be set to CPU count - 2.

    Raises
    -------
    ValueError

        - If ``file_paths`` is not a list.
        - If the application is unable to fetch any remote classes.
        - If the application was unable to find/parse any annotation files.
        - If the application was unable to fetch remote file list.

    IncompatibleOptions

        - If both ``append`` and ``delete_for_empty`` are specified as ``True``.
    """
    console = Console(theme=_console_theme())

    if append and delete_for_empty:
        raise IncompatibleOptions(
            "The options 'append' and 'delete_for_empty' cannot be used together. Use only one of them."
        )

    cpu_limit, use_multi_cpu = _get_multi_cpu_settings(
        cpu_limit, cpu_count(), use_multi_cpu
    )
    if use_multi_cpu:
        console.print(f"Using {cpu_limit} CPUs for parsing...", style="info")
    else:
        console.print("Using 1 CPU for parsing...", style="info")

    if not isinstance(file_paths, list):
        raise ValueError(
            f"file_paths must be a list of 'Path' or 'str'. Current value: {file_paths}"
        )

    console.print("Fetching remote class list...", style="info")
    team_classes: List[dt.DictFreeForm] = dataset.fetch_remote_classes(True)
    if not team_classes:
        raise ValueError("Unable to fetch remote class list.")

    if delete_for_empty and dataset.version == 1:
        console.print(
            f"The '--delete-for-empty' flag only works for V2 datasets. '{dataset.name}' is a V1 dataset. Ignoring flag.",
            style="warning",
        )

    classes_in_dataset: dt.DictFreeForm = build_main_annotations_lookup_table(
        [
            cls
            for cls in team_classes
            if cls["available"] or cls["name"] in GLOBAL_CLASSES
        ]
    )
    classes_in_team: dt.DictFreeForm = build_main_annotations_lookup_table(
        [
            cls
            for cls in team_classes
            if not cls["available"] and cls["name"] not in GLOBAL_CLASSES
        ]
    )
    attributes = build_attribute_lookup(dataset)

    console.print("Retrieving local annotations ...", style="info")
    local_files = []
    local_files_missing_remotely = []

    # ! Other place we can use multiprocessing - hard to pass in the importer though
    maybe_parsed_files: Optional[Iterable[dt.AnnotationFile]] = find_and_parse(
        importer, file_paths, console, use_multi_cpu, cpu_limit
    )

    if not maybe_parsed_files:
        raise ValueError("Not able to parse any files.")

    parsed_files: List[AnnotationFile] = flatten_list(list(maybe_parsed_files))

    filenames: List[str] = [
        parsed_file.filename for parsed_file in parsed_files if parsed_file is not None
    ]

    console.print("Fetching remote file list...", style="info")
    # This call will only filter by filename; so can return a superset of matched files across different paths
    # There is logic in this function to then include paths to narrow down to the single correct matching file
    remote_files: Dict[str, Tuple[int, str]] = {}

    # Try to fetch files in large chunks; in case the filenames are too large and exceed the url size
    # retry in smaller chunks
    chunk_size = 100
    while chunk_size > 0:
        try:
            remote_files = get_remote_files(dataset, filenames, chunk_size)
            break
        except RequestEntitySizeExceeded:
            chunk_size -= 8
            if chunk_size <= 0:
                raise ValueError("Unable to fetch remote file list.")

    for parsed_file in parsed_files:
        if parsed_file.full_path not in remote_files:
            local_files_missing_remotely.append(parsed_file)
        else:
            local_files.append(parsed_file)

    console.print(
        f"{len(local_files) + len(local_files_missing_remotely)} annotation file(s) found.",
        style="info",
    )
    if local_files_missing_remotely:
        console.print(
            f"{len(local_files_missing_remotely)} file(s) are missing from the dataset",
            style="warning",
        )
        for local_file in local_files_missing_remotely:
            console.print(
                f"\t{local_file.path}: '{local_file.full_path}'", style="warning"
            )

        if class_prompt and not secure_continue_request():
            return

    (
        local_classes_not_in_dataset,
        local_classes_not_in_team,
    ) = _resolve_annotation_classes(
        [
            annotation_class
            for file in local_files
            for annotation_class in file.annotation_classes
        ],
        classes_in_dataset,
        classes_in_team,
    )

    console.print(
        f"{len(local_classes_not_in_team)} classes needs to be created.", style="info"
    )
    console.print(
        f"{len(local_classes_not_in_dataset)} classes needs to be added to {dataset.identifier}",
        style="info",
    )

    missing_skeletons: List[dt.AnnotationClass] = list(
        filter(_is_skeleton_class, local_classes_not_in_team)
    )
    missing_skeleton_names: str = ", ".join(map(_get_skeleton_name, missing_skeletons))
    if missing_skeletons:
        console.print(
            f"Found missing skeleton classes: {missing_skeleton_names}. Missing Skeleton classes cannot be created. Exiting now.",
            style="error",
        )
        return

    if local_classes_not_in_team:
        console.print("About to create the following classes", style="info")
        for missing_class in local_classes_not_in_team:
            console.print(
                f"\t{missing_class.name}, type: {missing_class.annotation_internal_type or missing_class.annotation_type}",
                style="info",
            )
        if class_prompt and not secure_continue_request():
            return
        for missing_class in local_classes_not_in_team:
            dataset.create_annotation_class(
                missing_class.name,
                missing_class.annotation_internal_type or missing_class.annotation_type,
            )
    if local_classes_not_in_dataset:
        console.print(
            f"About to add the following classes to {dataset.identifier}", style="info"
        )
        for cls in local_classes_not_in_dataset:
            dataset.add_annotation_class(cls)

    # Refetch classes to update mappings
    if local_classes_not_in_team or local_classes_not_in_dataset:
        maybe_remote_classes: List[dt.DictFreeForm] = dataset.fetch_remote_classes()
        if not maybe_remote_classes:
            raise ValueError("Unable to fetch remote classes.")

        remote_classes = build_main_annotations_lookup_table(maybe_remote_classes)
    else:
        remote_classes = build_main_annotations_lookup_table(team_classes)

    if dataset.version == 1:
        console.print(
            "Importing annotations...\nEmpty annotations will be skipped.", style="info"
        )
    elif dataset.version == 2 and delete_for_empty:
        console.print(
            "Importing annotations...\nEmpty annotation file(s) will clear all existing annotations in matching remote files.",
            style="info",
        )
    else:
        console.print(
            "Importing annotations...\nEmpty annotations will be skipped, if you want to delete annotations rerun with '--delete-for-empty'.",
            style="info",
        )

    # Need to re parse the files since we didn't save the annotations in memory
    for local_path in set(local_file.path for local_file in local_files):  # noqa: C401
        imported_files: Union[
            List[dt.AnnotationFile], dt.AnnotationFile, None
        ] = importer(local_path)
        if imported_files is None:
            parsed_files = []
        elif not isinstance(imported_files, List):
            parsed_files = [imported_files]
        else:
            parsed_files = imported_files

        # remove files missing on the server
        missing_files = [
            missing_file.full_path for missing_file in local_files_missing_remotely
        ]
        parsed_files = [
            parsed_file
            for parsed_file in parsed_files
            if parsed_file.full_path not in missing_files
        ]

        files_to_not_track = [
            file_to_track
            for file_to_track in parsed_files
            if not file_to_track.annotations
            and (not delete_for_empty or dataset.version == 1)
        ]

        for file in files_to_not_track:
            console.print(
                f"{file.filename} has no annotations. Skipping upload...",
                style="warning",
            )

        files_to_track = [
            file for file in parsed_files if file not in files_to_not_track
        ]
        if files_to_track:
            _warn_unsupported_annotations(files_to_track)
            for parsed_file in track(files_to_track):
                image_id, default_slot_name = remote_files[parsed_file.full_path]
                # We need to check if name is not-None as Darwin JSON 1.0
                # defaults to name=None
                if parsed_file.slots and parsed_file.slots[0].name:
                    default_slot_name = parsed_file.slots[0].name

                metadata_path = is_properties_enabled(parsed_file.path)

                errors, _ = _import_annotations(
                    dataset.client,
                    image_id,
                    remote_classes,
                    attributes,
                    parsed_file.annotations,  # type: ignore
                    default_slot_name,
                    dataset,
                    append,
                    delete_for_empty,
                    import_annotators,
                    import_reviewers,
                    metadata_path,
                )

                if errors:
                    console.print(
                        f"Errors importing {parsed_file.filename}", style="error"
                    )
                    for error in errors:
                        console.print(f"\t{error}", style="error")


def _get_multi_cpu_settings(
    cpu_limit: Optional[int], cpu_count: int, use_multi_cpu: bool
) -> Tuple[int, bool]:
    if cpu_limit == 1 or cpu_count == 1 or not use_multi_cpu:
        return 1, False

    if cpu_limit is None:
        return max([cpu_count - 2, 2]), True

    return cpu_limit if cpu_limit <= cpu_count else cpu_count, True


def _warn_unsupported_annotations(parsed_files: List[AnnotationFile]) -> None:
    console = Console(theme=_console_theme())
    for parsed_file in parsed_files:
        skipped_annotations = []
        for annotation in parsed_file.annotations:
            if annotation.annotation_class.annotation_type in UNSUPPORTED_CLASSES:
                skipped_annotations.append(annotation)
        if len(skipped_annotations) > 0:
            types = {
                c.annotation_class.annotation_type for c in skipped_annotations
            }  # noqa: C417
            console.print(
                f"Import of annotation class types '{', '.join(types)}' is not yet supported. Skipping {len(skipped_annotations)} "
                + "annotations from '{parsed_file.full_path}'.\n",
                style="warning",
            )


def _is_skeleton_class(the_class: dt.AnnotationClass) -> bool:
    return (
        the_class.annotation_internal_type or the_class.annotation_type
    ) == "skeleton"


def _get_skeleton_name(skeleton: dt.AnnotationClass) -> str:
    return skeleton.name


def _handle_subs(
    annotation: dt.Annotation,
    data: dt.DictFreeForm,
    annotation_class_id: str,
    attributes: Dict[str, dt.UnknownType],
) -> dt.DictFreeForm:
    for sub in annotation.subs:
        if sub.annotation_type == "text":
            data["text"] = {"text": sub.data}
        elif sub.annotation_type == "attributes":
            attributes_with_key = []
            for attr in sub.data:
                if (
                    annotation_class_id in attributes
                    and attr in attributes[annotation_class_id]
                ):
                    attributes_with_key.append(attributes[annotation_class_id][attr])
                else:
                    print(
                        f"The attribute '{attr}' for class '{annotation.annotation_class.name}' was not imported."
                    )

            data["attributes"] = {"attributes": attributes_with_key}
        elif sub.annotation_type == "instance_id":
            data["instance_id"] = {"value": sub.data}
        else:
            data[sub.annotation_type] = sub.data

    return data


def _handle_complex_polygon(
    annotation: dt.Annotation, data: dt.DictFreeForm
) -> dt.DictFreeForm:
    if "complex_polygon" in data:
        del data["complex_polygon"]
        data["polygon"] = {
            "path": annotation.data["paths"][0],
            "additional_paths": annotation.data["paths"][1:],
        }
    return data


def _annotators_or_reviewers_to_payload(
    actors: List[dt.AnnotationAuthor], role: dt.AnnotationAuthorRole
) -> List[dt.DictFreeForm]:
    return [{"email": actor.email, "role": role.value} for actor in actors]


def _handle_reviewers(
    annotation: dt.Annotation, import_reviewers: bool
) -> List[dt.DictFreeForm]:
    if import_reviewers:
        if annotation.reviewers:
            return _annotators_or_reviewers_to_payload(
                annotation.reviewers, dt.AnnotationAuthorRole.REVIEWER
            )
    return []


def _handle_annotators(
    annotation: dt.Annotation, import_annotators: bool
) -> List[dt.DictFreeForm]:
    if import_annotators:
        if annotation.annotators:
            return _annotators_or_reviewers_to_payload(
                annotation.annotators, dt.AnnotationAuthorRole.ANNOTATOR
            )
    return []


def _get_annotation_data(
    annotation: dt.AnnotationLike, annotation_class_id: str, attributes: dt.DictFreeForm
) -> dt.DictFreeForm:
    annotation_class = annotation.annotation_class
    if isinstance(annotation, dt.VideoAnnotation):
        data = annotation.get_data(
            only_keyframes=True,
            post_processing=lambda annotation, data: \
                _handle_subs(
                    annotation,
                    _handle_complex_polygon(annotation, data),
                    annotation_class_id,
                    attributes,
                ),
        )
    else:
        data = {annotation_class.annotation_type: annotation.data}
        data = _handle_complex_polygon(annotation, data)
        data = _handle_subs(annotation, data, annotation_class_id, attributes)

    return data


def _handle_slot_names(
    annotation: dt.Annotation, dataset_version: int, default_slot_name: str
) -> dt.Annotation:
    if not annotation.slot_names and dataset_version > 1:
        annotation.slot_names.extend([default_slot_name])

    return annotation


def _get_overwrite_value(append: bool) -> str:
    return "false" if append else "true"


def _import_annotations(
    client: "Client",  # TODO: This is unused, should it be?
    id: Union[str, int],
    remote_classes: dt.DictFreeForm,
    attributes: dt.DictFreeForm,
    annotations: List[dt.Annotation],
    default_slot_name: str,
    dataset: "RemoteDataset",
    append: bool,
    delete_for_empty: bool,  # TODO: This is unused, should it be?
    import_annotators: bool,
    import_reviewers: bool,
    metadata_path: Union[Path, bool] = False,
) -> Tuple[dt.ErrorList, dt.Success]:
    errors: dt.ErrorList = []
    success: dt.Success = dt.Success.SUCCESS

    rl: Optional[dt.Annotation] = None
    rl_dense_rle_ids: Optional[Set[str]] = None
    serialized_annotations = []
    annotation_class_ids_map: Dict[str, str] = {}
    for annotation in annotations:
        annotation_class = annotation.annotation_class
        annotation_type = (
            annotation_class.annotation_internal_type
            or annotation_class.annotation_type
        )

        if (
            annotation_type not in remote_classes
            or annotation_class.name not in remote_classes[annotation_type]
            and annotation_type
            != "raster_layer"  # We do not skip raster layers as they are always available.
        ):
            if annotation_type not in remote_classes:
                logger.warning(
                    f"Annotation type '{annotation_type}' is not in the remote classes, skipping import of annotation '{annotation_class.name}'"
                )
            else:
                logger.warning(
                    f"Annotation '{annotation_class.name}' is not in the remote classes, skipping import"
                )
            continue

        annotation_class_id: str = remote_classes[annotation_type][
            annotation_class.name
        ]

        data = _get_annotation_data(annotation, annotation_class_id, attributes)

        # check if the mask is empty (i.e. masks that do not have a corresponding raster layer) if so, skip import of the mask
        if annotation_type == "mask":
            if rl is None:
                rl = next(
                    (
                        a
                        for a in annotations
                        if a.annotation_class.annotation_type == "raster_layer"
                    ),
                    None,
                )
            if rl and rl_dense_rle_ids is None:
                rl_dense_rle_ids = set(rl.data["dense_rle"][::2])
            if rl is not None and rl_dense_rle_ids is not None:
                # check if the 'annotation_class_id' is in raster_layer's mask_annotation_ids_mapping dict
                _annotation_id = annotation.id
                if (
                    rl.data["mask_annotation_ids_mapping"][_annotation_id]
                    not in rl_dense_rle_ids
                ):
                    # skip import of the mask, and remove it from mask_annotation_ids_mapping
                    logger.warning(
                        f"Skipping import of mask annotation '{annotation_class.name}' as it does not have a corresponding raster layer"
                    )
                    del rl.data["mask_annotation_ids_mapping"][_annotation_id]
                    continue

        actors: List[dt.DictFreeForm] = []
        actors.extend(_handle_annotators(annotation, import_annotators))
        actors.extend(_handle_reviewers(annotation, import_reviewers))

        # Insert the default slot name if not available in the import source
        annotation = _handle_slot_names(annotation, dataset.version, default_slot_name)

        annotation_class_ids_map[annotation_class.name] = annotation_class_id
        serial_obj = {
            "annotation_class_id": annotation_class_id,
            "data": data,
            "context_keys": {"slot_names": annotation.slot_names},
        }

        if annotation.id:
            serial_obj["id"] = annotation.id

        if actors:
            serial_obj["actors"] = actors  # type: ignore

        serialized_annotations.append(serial_obj)

    annotation_id_property_map = _import_properties(
        metadata_path,
        client,
        annotations,  # type: ignore
        annotation_class_ids_map,
    )
    _update_payload_with_properties(
        serialized_annotations,
        annotation_id_property_map
    )

    payload: dt.DictFreeForm = {"annotations": serialized_annotations}
    payload["overwrite"] = _get_overwrite_value(append)

    try:
        dataset.import_annotation(id, payload=payload)
    except Exception as e:
        errors.append(e)
        success = dt.Success.FAILURE

    return errors, success


# mypy: ignore-errors
def _console_theme() -> Theme:
    return Theme(
        {
            "success": "bold green",
            "warning": "bold yellow",
            "error": "bold red",
            "info": "bold deep_sky_blue1",
        }
    )
