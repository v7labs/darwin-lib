import concurrent.futures
import uuid
import copy
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

import json
import numpy as np
import nibabel as nib
from darwin.datatypes import (
    AnnotationFile,
    Property,
    parse_property_classes,
    PropertyClass,
)
from darwin.future.data_objects.properties import (
    FullProperty,
    PropertyType,
    PropertyValue,
    SelectedProperty,
    PropertyGranularity,
)
from darwin.item import DatasetItem
from darwin.path_utils import is_properties_enabled, parse_metadata
from darwin.utils.utils import _parse_annotators

Unknown = Any  # type: ignore
from tqdm import tqdm

if TYPE_CHECKING:
    from darwin.client import Client
    from darwin.dataset.remote_dataset import RemoteDataset

from rich.console import Console
from rich.theme import Theme

import darwin.datatypes as dt
from darwin.datatypes import PathLike
from darwin.exceptions import IncompatibleOptions, RequestEntitySizeExceeded
from darwin.utils import secure_continue_request
from darwin.utils.flatten_list import flatten_list

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

# Add a length of 5 for URL encoding overhead and separators per filename
FILENAME_OVERHEAD = 5
MAX_URL_LENGTH = 2000
BASE_URL_LENGTH = 200

DEPRECATION_MESSAGE = """

This function is going to be turned into private. This means that breaking
changes in its interface and implementation are to be expected. We encourage using ``import_annotations``
instead of calling this low-level function directly.

"""


def _build_main_annotations_lookup_table(
    annotation_classes: List[Dict[str, Unknown]],
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
        "simple_table",
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


def _find_and_parse(  # noqa: C901
    importer: Callable[[Path], Union[List[dt.AnnotationFile], dt.AnnotationFile, None]],
    file_paths: List[PathLike],
    console: Optional[Console] = None,
    use_multi_cpu: bool = True,
    cpu_limit: int = 1,
    remote_file_medical_metadata: Optional[Dict[Path, Dict[str, Any]]] = {},
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
                if importer.__module__ == "darwin.importer.formats.nifti":
                    parsed_files = pool.map(
                        lambda file: importer(
                            file,
                            remote_file_medical_metadata=remote_file_medical_metadata,
                        ),
                        tqdm(files),
                    )
                else:
                    parsed_files = pool.map(
                        importer, tqdm(files) if is_console else files
                    )
        except KeyboardInterrupt:
            maybe_console("Keyboard interrupt. Stopping.")
            return None
        except Exception as e:
            maybe_console(f"Error: {e}")
            return None

    else:
        maybe_console("Using single CPU")
        if importer.__module__ == "darwin.importer.formats.nifti":
            parsed_files = [
                importer(
                    file,
                    remote_file_medical_metadata=remote_file_medical_metadata,
                )
                for file in tqdm(files)
            ]
        else:
            parsed_files = [
                importer(file) for file in (tqdm(files) if is_console else files)
            ]
    parsed_files = [f for f in parsed_files if f is not None]

    maybe_console("Finished.")
    # Sometimes we have a list of lists of AnnotationFile, sometimes we have a list of AnnotationFile
    # We flatten the list of lists
    if not parsed_files:
        return None
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


def _build_attribute_lookup(dataset: "RemoteDataset") -> Dict[str, Unknown]:
    attributes: List[Dict[str, Unknown]] = dataset.fetch_remote_attributes()
    lookup: Dict[str, Unknown] = {}
    for attribute in attributes:
        class_id = attribute["class_id"]
        if class_id not in lookup:
            lookup[class_id] = {}
        lookup[class_id][attribute["name"]] = attribute["id"]
    return lookup


def _get_remote_files_ready_for_import(
    dataset: "RemoteDataset",
    filenames: List[str],
    chunk_size: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetches remote files that are ready for import from the datasets in chunks; by
    default 100 filenames at a time.

    The output is a dictionary for each remote file with the following keys:
    - "item_id": Item ID
    - "slot_names": A list of each slot name for the item
    - "layout": The layout of the item

    Fetching slot names & layout is necessary here to avoid double-trip to API downstream for remote files.

    Raises a ValueError if any of the remote files are not in the `new`, `annotate`,
    `review`, `complete`, or `archived` statuses.

    Parameters
    ----------
    dataset : RemoteDataset
        The remote dataset to fetch the files from.
    filenames : List[str]
        A list of filenames to fetch.
    chunk_size : int
        The number of filenames to fetch at a time.
    """
    remote_files = {}
    remote_files_not_ready_for_import = {}
    for i in range(0, len(filenames), chunk_size):
        chunk = filenames[i : i + chunk_size]
        for remote_file in dataset.fetch_remote_files(
            {"types": "image,playback_video,video_frame", "item_names": chunk}
        ):
            if remote_file.status not in [
                "new",
                "annotate",
                "review",
                "complete",
                "archived",
            ]:
                remote_files_not_ready_for_import[remote_file.full_path] = (
                    remote_file.status
                )
            else:
                slot_names = _get_slot_names(remote_file)
                remote_files[remote_file.full_path] = {
                    "item_id": remote_file.id,
                    "slot_names": slot_names,
                    "layout": remote_file.layout,
                }
    if remote_files_not_ready_for_import:
        console = Console(theme=_console_theme())
        console.print(
            "The following files are either still processing, or failed to process, so annotations cannot be imported:",
            style="warning",
        )
        for file, status in remote_files_not_ready_for_import.items():
            console.print(f"  - {file}, status: {status}")
        raise ValueError(
            "Some files targeted for annotation import are either still processing, or failed to process, so annotations cannot be imported."
        )
    return remote_files


def _get_slot_names(remote_file: DatasetItem) -> List[str]:
    """
    Returns a list of slot names for a dataset item:
    - If the item's layout is V1 or V2, it is multi-slotted.
      In this case we return the slot names in the order they appear in `slots`.
      This ensures that the default slot is the first item in the list
    - If the item's layout is V3, it is multi-channeled.
      In this case we return the slot names in the order they appear in `slots_grid`.
      This ensures that the base slot is the first item in the list

    Parameters
    ----------
    remote_file : DatasetItem
        A DatasetItem object representing a single remote dataset item

    Returns
    -------
    List[str]
        A list of slot names associated with the item
    """
    layout_version = remote_file.layout["version"]
    if layout_version == 1 or layout_version == 2:
        return [slot["slot_name"] for slot in remote_file.slots]
    elif layout_version == 3:
        return list(remote_file.layout["slots_grid"][0][0])


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


def _get_team_properties_annotation_lookup(
    client: "Client", team_slug: str
) -> Tuple[Dict[Tuple[str, Optional[int]], FullProperty], Dict[str, FullProperty]]:
    """
    Returns two lookup dictionaries for team properties:
     - team_properties_annotation_lookup: (property-name, annotation_class_id): FullProperty object
     - team_item_properties_lookup: property-name: FullProperty object

    Args:
        client (Client): Darwin Client object
        team_slug (str): Team slug

    Returns:
        Tuple[Dict[Tuple[str, Optional[int]], FullProperty], Dict[str, FullProperty]: Tuple of two dictionaries
    """
    # get team properties -> List[FullProperty]
    team_properties = client.get_team_properties(team_slug)

    # (property-name, annotation_class_id): FullProperty object
    team_properties_annotation_lookup: Dict[Tuple[str, Optional[int]], FullProperty] = (
        {}
    )

    # property-name: FullProperty object
    team_item_properties_lookup: Dict[str, FullProperty] = {}
    for prop in team_properties:
        if (
            prop.granularity.value == "section"
            or prop.granularity.value == "annotation"
        ):
            team_properties_annotation_lookup[(prop.name, prop.annotation_class_id)] = (
                prop
            )
        elif prop.granularity.value == "item":
            team_item_properties_lookup[prop.name] = prop

    return team_properties_annotation_lookup, team_item_properties_lookup


def _update_payload_with_properties(
    annotations: List[Dict[str, Unknown]],
    annotation_id_property_map: Dict[str, Dict[str, Dict[str, Set[str]]]],
) -> None:
    """
    Updates the annotations with the properties that were created/updated during the import.

    Args:
        annotations (List[dt.Annotation]): List of annotations
        annotation_id_property_map: Dict[str, Dict[str, Dict[str, Set[str]]]]: Dict of annotation.id to frame_index -> property id -> property val ids
    """
    if not annotation_id_property_map:
        return

    for annotation in annotations:
        annotation_id = annotation["id"]

        if annotation_id_property_map.get(annotation_id):
            _map = {}
            for _frame_index, _property_map in annotation_id_property_map[
                annotation_id
            ].items():
                _map[_frame_index] = {}
                for prop_id, prop_val_set in dict(_property_map).items():
                    prop_val_list = list(prop_val_set)
                    _map[_frame_index][prop_id] = prop_val_list

            annotation["annotation_properties"] = dict(_map)


def _serialize_item_level_properties(
    item_property_values: List[Dict[str, str]],
    client: "Client",
    dataset: "RemoteDataset",
    import_annotators: bool,
    import_reviewers: bool,
) -> List[Dict[str, Any]]:
    """
    Returns serialized item-level properties to be added to the annotation import payload.

    Args:
        item_property_values (List[Dict[str, str]]): A list of dictionaries containing item property values.
        client (Client): The client instance used to interact with the API.
        dataset (RemoteDataset): The remote dataset instance.
        import_annotators (bool): Flag indicating whether to import annotators.
        import_reviewers (bool): Flag indicating whether to import reviewers.

    Returns:
        List[Dict[str, Any]]: A list of serialized item-level properties for the annotation import payload.
    """
    if not item_property_values:
        return []

    serialized_item_level_properties: List[Dict[str, Any]] = []
    actors: List[dt.DictFreeForm] = []
    # Get team properties
    _, team_item_properties_lookup = _get_team_properties_annotation_lookup(
        client, dataset.team
    )
    for item_property_value in item_property_values:
        item_property = team_item_properties_lookup[item_property_value["name"]]
        item_property_id = item_property.id
        if (
            item_property.type == "single_select"
            or item_property.type == "multi_select"
        ):
            item_property_value_id = next(
                (
                    pv.id
                    for pv in item_property.property_values or []
                    if pv.value == item_property_value["value"]
                ),
                None,
            )
            value = {"id": item_property_value_id}
        elif item_property.type == "text":
            value = {"text": item_property_value["value"]}
        actors: List[dt.DictFreeForm] = []
        actors.extend(
            _handle_annotators(
                import_annotators, item_property_value=item_property_value
            )
        )
        actors.extend(
            _handle_reviewers(import_reviewers, item_property_value=item_property_value)
        )
        serialized_item_level_properties.append(
            {
                "actors": actors,
                "property_id": item_property_id,
                "value": value,
            }
        )

    return serialized_item_level_properties


def _parse_metadata_file(
    metadata_path: Union[Path, bool],
) -> Tuple[List[PropertyClass], List[Dict[str, str]]]:
    if isinstance(metadata_path, Path):
        metadata = parse_metadata(metadata_path)
        metadata_property_classes = parse_property_classes(metadata)
        metadata_item_props = metadata.get("properties", [])
        return metadata_property_classes, metadata_item_props
    return [], []


def _build_metadata_lookups(
    metadata_property_classes: List[PropertyClass],
    metadata_item_props: List[Dict[str, str]],
) -> Tuple[
    Set[Tuple[str, str]],
    Dict[Tuple[str, str], Property],
    Dict[Tuple[int, str], Property],
    Dict[str, Property],
]:
    metadata_classes_lookup = set()
    metadata_cls_prop_lookup = {}
    metadata_cls_id_prop_lookup = {}
    metadata_item_prop_lookup = {}

    for _cls in metadata_property_classes:
        metadata_classes_lookup.add((_cls.name, _cls.type))
        for _prop in _cls.properties or []:
            metadata_cls_prop_lookup[(_cls.name, _prop.name)] = _prop
    for _item_prop in metadata_item_props:
        metadata_item_prop_lookup[_item_prop["name"]] = _item_prop

    return (
        metadata_classes_lookup,
        metadata_cls_prop_lookup,
        metadata_cls_id_prop_lookup,
        metadata_item_prop_lookup,
    )


def _import_properties(
    metadata_path: Union[Path, bool],
    item_properties: List[Dict[str, str]],
    client: "Client",
    annotations: List[dt.Annotation],
    annotation_class_ids_map: Dict[Tuple[str, str], str],
    dataset: "RemoteDataset",
) -> Dict[str, Dict[str, Dict[str, Set[str]]]]:
    """
    Creates/Updates missing/mismatched properties from annotation & metadata.json file to team-properties.
    As the properties are created/updated, the annotation_id_property_map is updated with the new/old property ids.
    ^ This is used in the import-annotations payload later on.

    Args:
        metadata_path (Union[Path, bool]): Path object to .v7/metadata.json file
        client (Client): Darwin Client object
        item_properties (List[Dict[str, str]]): List of item-level properties present in the annotation file
        annotations (List[dt.Annotation]): List of annotations
        annotation_class_ids_map (Dict[Tuple[str, str], str]): Dict of annotation class names/types to annotation class ids
        dataset (RemoteDataset): RemoteDataset object

    Raises:
        ValueError: raise error if annotation class not present in metadata and in team-properties
        ValueError: raise error if annotation-property not present in metadata and in team-properties
        ValueError: raise error if property value is missing for a property that requires a value
        ValueError: raise error if property value/type is different in m_prop (.v7/metadata.json) options

    Returns:
        Dict[str, Dict[str, Dict[str, Set[str]]]]: Dict of annotation.id to frame_index -> property id -> property val ids
    """
    annotation_property_map: Dict[str, Dict[str, Dict[str, Set[str]]]] = {}

    # Parse metadata
    metadata_property_classes, metadata_item_props = _parse_metadata_file(metadata_path)

    # Get team properties
    (
        team_properties_annotation_lookup,
        team_item_properties_lookup,
    ) = _get_team_properties_annotation_lookup(client, dataset.team)

    # Build metadata lookups
    (
        metadata_classes_lookup,
        metadata_cls_prop_lookup,
        metadata_cls_id_prop_lookup,
        metadata_item_prop_lookup,
    ) = _build_metadata_lookups(metadata_property_classes, metadata_item_props)

    # (annotation-id): dt.Annotation object
    annotation_id_map: Dict[str, dt.Annotation] = {}

    annotation_and_section_level_properties_to_create: List[FullProperty] = []
    annotation_and_section_level_properties_to_update: List[FullProperty] = []
    for annotation in annotations:
        annotation_name = annotation.annotation_class.name
        annotation_type = annotation_type = (
            annotation.annotation_class.annotation_internal_type
            or annotation.annotation_class.annotation_type
        )
        annotation_name_type = (annotation_name, annotation_type)
        if annotation_name_type not in annotation_class_ids_map:
            continue
        annotation_class_id = int(annotation_class_ids_map[annotation_name_type])
        if not annotation.id:
            continue
        annotation_id = annotation.id
        if annotation_id not in annotation_property_map:
            annotation_property_map[annotation_id] = defaultdict(
                lambda: defaultdict(set)
            )
        annotation_id_map[annotation_id] = annotation

        # loop on annotation properties and check if they exist in metadata & team
        for a_prop in annotation.properties or []:
            a_prop: SelectedProperty

            # raise error if annotation-property not present in metadata
            if (annotation_name, a_prop.name) not in metadata_cls_prop_lookup:
                # check if they are present in team properties
                if (
                    a_prop.name,
                    annotation_class_id,
                ) in team_properties_annotation_lookup:
                    # get team property
                    t_prop: FullProperty = team_properties_annotation_lookup[
                        (a_prop.name, annotation_class_id)
                    ]

                    # if property value is None, update annotation_property_map with empty set
                    if a_prop.value is None:
                        assert t_prop.id is not None

                        annotation_property_map[annotation_id][str(a_prop.frame_index)][
                            t_prop.id
                        ] = set()
                        continue

                    # get team property value
                    t_prop_val = None
                    for prop_val in t_prop.property_values or []:
                        if prop_val.value == a_prop.value:
                            t_prop_val = prop_val
                            break

                    # if property value exists in team properties, update annotation_property_map
                    if t_prop_val:
                        assert t_prop.id is not None
                        assert t_prop_val.id is not None
                        annotation_property_map[annotation_id][str(a_prop.frame_index)][
                            t_prop.id
                        ].add(t_prop_val.id)
                        continue

                # TODO: Change this so that if a property isn't found in the metadata, we can create it assuming it's an option, multi-select with no description (DAR-2920)
                raise ValueError(
                    f"Annotation: '{annotation_name}' -> Property '{a_prop.name}' not found in {metadata_path}"
                )

            # get metadata property
            m_prop: Property = metadata_cls_prop_lookup[(annotation_name, a_prop.name)]

            # update metadata-property lookup
            metadata_cls_id_prop_lookup[(annotation_class_id, a_prop.name)] = m_prop

            # get metadata property type
            m_prop_type: PropertyType = m_prop.type

            # get metadata property options
            m_prop_options: List[Dict[str, str]] = m_prop.property_values or []

            # check if property value is missing for a property that requires a value
            if m_prop.required and not a_prop.value:
                raise ValueError(
                    f"Annotation: '{annotation_name}' -> Property '{a_prop.name}' requires a value!"
                )

            # check if property and annotation class exists in team
            if (
                a_prop.name,
                annotation_class_id,
            ) not in team_properties_annotation_lookup:
                # check if fullproperty exists in annotation_and_section_level_properties_to_create
                for full_property in annotation_and_section_level_properties_to_create:
                    if (
                        full_property.name == a_prop.name
                        and full_property.annotation_class_id == annotation_class_id
                    ):
                        # make sure property_values is not None
                        if full_property.property_values is None:
                            full_property.property_values = []

                        property_values = full_property.property_values
                        if a_prop.value is None:
                            # skip creating property if property value is None
                            continue
                        # find property value in m_prop (.v7/metadata.json) options
                        for m_prop_option in m_prop_options:
                            if m_prop_option.get("value") == a_prop.value:
                                # check if property value exists in property_values
                                for prop_val in property_values:
                                    if prop_val.value == a_prop.value:
                                        break
                                else:
                                    # update property_values with new value
                                    full_property.property_values.append(
                                        PropertyValue(
                                            value=m_prop_option.get("value"),  # type: ignore
                                            color=m_prop_option.get("color"),  # type: ignore
                                        )
                                    )
                                break
                        break
                else:
                    property_values = []
                    if a_prop.value is None:
                        # skip creating property if property value is None
                        continue
                    # find property value in m_prop (.v7/metadata.json) options
                    for m_prop_option in m_prop_options:
                        if m_prop_option.get("value") == a_prop.value:
                            property_values.append(
                                PropertyValue(
                                    value=m_prop_option.get("value"),  # type: ignore
                                    color=m_prop_option.get("color"),  # type: ignore
                                )
                            )
                            break
                    # if it doesn't exist, create it
                    for prop in annotation_and_section_level_properties_to_create:
                        if (
                            prop.name == a_prop.name
                            and prop.annotation_class_id == annotation_class_id
                        ):
                            current_prop_values = [
                                value.value for value in prop.property_values
                            ]
                            for value in property_values:
                                if value.value not in current_prop_values:
                                    prop.property_values.append(value)
                            break
                    else:
                        full_property = FullProperty(
                            name=a_prop.name,
                            type=m_prop_type,  # type from .v7/metadata.json
                            required=m_prop.required,  # required from .v7/metadata.json
                            description=m_prop.description
                            or "property-created-during-annotation-import",
                            slug=client.default_team,
                            annotation_class_id=int(annotation_class_id),
                            property_values=property_values,
                            granularity=PropertyGranularity(m_prop.granularity),
                        )
                    # Don't attempt the same propery creation multiple times
                    if (
                        full_property
                        not in annotation_and_section_level_properties_to_create
                    ):
                        annotation_and_section_level_properties_to_create.append(
                            full_property
                        )
                continue

            # check if property value is different in m_prop (.v7/metadata.json) options
            for m_prop_option in m_prop_options:
                if m_prop_option.get("value") == a_prop.value:
                    break
            else:
                if a_prop.value:
                    raise ValueError(
                        f"Annotation: '{annotation_name}' -> Property '{a_prop.value}' not found in .v7/metadata.json, found: {m_prop.property_values}"
                    )

            # get team property
            t_prop: FullProperty = team_properties_annotation_lookup[
                (a_prop.name, annotation_class_id)
            ]

            if a_prop.value is None:
                # if property value is None, update annotation_property_map with empty set
                assert t_prop.id is not None
                annotation_property_map[annotation_id][str(a_prop.frame_index)][
                    t_prop.id
                ] = set()
                continue

            # check if property value is different in t_prop (team) options
            for t_prop_val in t_prop.property_values or []:
                if t_prop_val.value == a_prop.value:
                    break
            else:
                # if it is, update it
                full_property = FullProperty(
                    id=t_prop.id,
                    name=a_prop.name,
                    type=m_prop_type,
                    required=m_prop.required,
                    description=m_prop.description
                    or "property-updated-during-annotation-import",
                    slug=client.default_team,
                    annotation_class_id=int(annotation_class_id),
                    property_values=[
                        PropertyValue(
                            value=a_prop.value,
                            color=m_prop_option.get("color"),  # type: ignore
                        )
                    ],
                    granularity=t_prop.granularity,
                )
                # Don't attempt the same propery update multiple times
                if (
                    full_property
                    not in annotation_and_section_level_properties_to_update
                ):
                    annotation_and_section_level_properties_to_update.append(
                        full_property
                    )
                continue

            assert t_prop.id is not None
            assert t_prop_val.id is not None
            annotation_property_map[annotation_id][str(a_prop.frame_index)][
                t_prop.id
            ].add(t_prop_val.id)

    # Create/Update team item properties based on metadata
    (
        item_properties_to_create_from_metadata,
        item_properties_to_update_from_metadata,
    ) = _create_update_item_properties(
        _normalize_item_properties(metadata_item_prop_lookup),
        team_item_properties_lookup,
        client,
    )

    console = Console(theme=_console_theme())

    properties_to_create = (
        annotation_and_section_level_properties_to_create
        + item_properties_to_create_from_metadata
    )
    properties_to_update = (
        annotation_and_section_level_properties_to_update
        + item_properties_to_update_from_metadata
    )

    created_properties = []
    if properties_to_create:
        console.print(f"Creating {len(properties_to_create)} properties:", style="info")
        for full_property in properties_to_create:
            if full_property.granularity.value == "item":
                console.print(
                    f"- Creating item-level property '{full_property.name}' of type: {full_property.type}"
                )
            else:
                console.print(
                    f"- Creating property '{full_property.name}' of type {full_property.type}",
                )
            prop = client.create_property(
                team_slug=full_property.slug, params=full_property
            )
            created_properties.append(prop)

    updated_properties = []
    if properties_to_update:
        console.print(
            f"Performing {len(properties_to_update)} property update(s):", style="info"
        )
        for full_property in properties_to_update:
            if full_property.granularity.value == "item":
                console.print(
                    f"- Updating item-level property '{full_property.name}' with new value: {full_property.property_values[0].value}",
                )
            else:
                console.print(
                    f"- Updating property '{full_property.name}' of type {full_property.type}",
                )
            prop = client.update_property(
                team_slug=full_property.slug, params=full_property
            )
            updated_properties.append(prop)

    # get latest team properties
    (
        team_properties_annotation_lookup,
        team_item_properties_lookup,
    ) = _get_team_properties_annotation_lookup(client, dataset.team)

    # Update item-level properties from annotations
    _, item_properties_to_update_from_annotations = _create_update_item_properties(
        _normalize_item_properties(item_properties),
        team_item_properties_lookup,
        client,
    )

    if item_properties_to_update_from_annotations:
        console.print(
            f"Performing {len(item_properties_to_update_from_annotations)} property update(s):",
            style="info",
        )
        for full_property in item_properties_to_update_from_annotations:
            if full_property.granularity.value == "item":
                console.print(
                    f"- Updating item-level property '{full_property.name}' with new value: {full_property.property_values[0].value}"
                )
            else:
                console.print(
                    f"- Updating property {full_property.name} ({full_property.type})",
                )
            prop = client.update_property(
                team_slug=full_property.slug, params=full_property
            )
            updated_properties.append(prop)

    # get latest team properties
    (
        team_properties_annotation_lookup,
        team_item_properties_lookup,
    ) = _get_team_properties_annotation_lookup(client, dataset.team)

    # loop over metadata_cls_id_prop_lookup, and update additional metadata property values
    for (annotation_class_id, prop_name), m_prop in metadata_cls_id_prop_lookup.items():
        # does the annotation-property exist in the team? if not, skip
        if (prop_name, annotation_class_id) not in team_properties_annotation_lookup:
            continue

        # get metadata property values
        m_prop_values = {
            m_prop_val["value"]: m_prop_val
            for m_prop_val in m_prop.property_values or []
            if m_prop_val["value"]
        }

        # get team property
        t_prop: FullProperty = team_properties_annotation_lookup[
            (prop_name, annotation_class_id)
        ]

        # get team property values
        t_prop_values = [prop_val.value for prop_val in t_prop.property_values or []]

        # get diff of metadata property values and team property values
        extra_values = set(m_prop_values.keys()) - set(t_prop_values)

        # if there are extra values in metadata, create a new FullProperty with the extra values
        if extra_values:
            extra_property_values = [
                PropertyValue(
                    value=m_prop_values[extra_value].get("value"),  # type: ignore
                    color=m_prop_values[extra_value].get("color"),  # type: ignore
                )
                for extra_value in extra_values
            ]
            full_property = FullProperty(
                id=t_prop.id,
                name=t_prop.name,
                type=t_prop.type,
                required=t_prop.required,
                description=t_prop.description,
                slug=client.default_team,
                annotation_class_id=t_prop.annotation_class_id,
                property_values=extra_property_values,
                granularity=PropertyGranularity(t_prop.granularity.value),
            )
            console.print(
                f"Updating property {full_property.name} ({full_property.type}) with extra metadata values {extra_values}",
                style="info",
            )
            prop = client.update_property(
                team_slug=full_property.slug, params=full_property
            )

    # update annotation_property_map with property ids from created_properties & updated_properties
    for annotation_id, _ in annotation_property_map.items():
        if not annotation_id_map.get(annotation_id):
            continue
        annotation = annotation_id_map[annotation_id]
        annotation_class = annotation.annotation_class
        annotation_class_name = annotation_class.name
        annotation_type = (
            annotation_class.annotation_internal_type
            or annotation_class.annotation_type
        )
        annotation_class_id = annotation_class_ids_map[
            (annotation_class_name, annotation_type)
        ]
        for a_prop in annotation.properties or []:
            frame_index = str(a_prop.frame_index)

            for prop in created_properties + updated_properties:
                if (
                    prop.name == a_prop.name
                    and annotation_class_id == prop.annotation_class_id
                ):
                    if a_prop.value is None:
                        if not annotation_property_map[annotation_id][frame_index][
                            prop.id
                        ]:
                            annotation_property_map[annotation_id][frame_index][
                                prop.id
                            ] = set()
                            break

                    for prop_val in prop.property_values or []:
                        if prop_val.value == a_prop.value:
                            annotation_property_map[annotation_id][frame_index][
                                prop.id
                            ].add(prop_val.id)
                            break
                    break
    _assign_item_properties_to_dataset(
        item_properties, team_item_properties_lookup, client, dataset, console
    )

    return annotation_property_map


def _normalize_item_properties(
    item_properties: Union[Dict[str, Dict[str, Any]], List[Dict[str, str]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Normalizes item properties to a common dictionary format.

    Args:
        item_properties (Union[Dict[str, Dict[str, Any]], List[Dict[str, str]]]): Item properties in different formats.

    Returns:
        Dict[str, Dict[str, Any]]: Normalized item properties.
    """
    if isinstance(item_properties, dict):
        return item_properties

    normalized_properties = defaultdict(lambda: {"property_values": []})
    if item_properties:
        for item_prop in item_properties:
            name = item_prop["name"]
            value = item_prop["value"]
            if value:
                normalized_properties[name]["property_values"].append({"value": value})

    return normalized_properties


def _create_update_item_properties(
    item_properties: Dict[str, Dict[str, Any]],
    team_item_properties_lookup: Dict[str, FullProperty],
    client: "Client",
) -> Tuple[List[FullProperty], List[FullProperty]]:
    """
    Compares item-level properties present in `item_properties` with the team item properties and plans to create or update them.

    Args:
        item_properties (Dict[str, Dict[str, Any]]): Dictionary of item-level properties present in the annotation file
        team_item_properties_lookup (Dict[str, FullProperty]): Lookup of team item properties
        client (Client): Darwin Client object

    Returns:
        Tuple[List[FullProperty], List[FullProperty]]: Tuple of lists of properties to be created and updated
    """
    create_properties = []
    update_properties = []
    for item_prop_name, m_prop in item_properties.items():
        m_prop_values = [
            prop_val["value"] for prop_val in m_prop.get("property_values", [])
        ]

        # If the property exists in the team, check that all values are present
        if item_prop_name in team_item_properties_lookup:
            t_prop = team_item_properties_lookup[item_prop_name]
            # If the property is a text property it won't have predefined values, so continue
            if t_prop.type == "text":
                continue
            t_prop_values = [
                prop_val.value for prop_val in t_prop.property_values or []
            ]

            # Add one update per missing property value
            for m_prop_value in m_prop_values:
                if m_prop_value not in t_prop_values:
                    update_property = FullProperty(
                        id=t_prop.id,
                        name=t_prop.name,
                        type=t_prop.type,
                        required=t_prop.required,
                        description=t_prop.description,
                        slug=client.default_team,
                        annotation_class_id=t_prop.annotation_class_id,
                        property_values=[PropertyValue(value=m_prop_value)],
                        granularity=PropertyGranularity.item,
                    )
                    update_properties.append(update_property)

        # If the property does not exist in the team, create it
        else:
            # If we've already planned to create this property, simply extend the property values
            for prop in create_properties:
                if prop.name == item_prop_name:
                    current_prop_values = [
                        value.value for value in prop.property_values
                    ]
                    if prop.property_values is None:
                        prop.property_values = []
                    for val in m_prop_values:
                        if val.value not in current_prop_values:
                            prop.property_values.append(PropertyValue(value=val))
                    break
            else:
                full_property = FullProperty(
                    name=item_prop_name,
                    type=m_prop.get("type", "multi_select"),
                    required=bool(m_prop.get("required", False)),
                    description=m_prop.get(
                        "description", "property-created-during-annotation-import"
                    ),
                    slug=client.default_team,
                    annotation_class_id=None,
                    property_values=[PropertyValue(value=val) for val in m_prop_values],
                    granularity=PropertyGranularity.item,
                )
                create_properties.append(full_property)

    return create_properties, update_properties


def _assign_item_properties_to_dataset(
    item_properties: List[Dict[str, str]],
    team_item_properties_lookup: Dict[str, FullProperty],
    client: "Client",
    dataset: "RemoteDataset",
    console: Console,
) -> None:
    """
    Ensures that all item-level properties to be imported are assigned to the target dataset

    Args:
        item_properties (List[Dict[str, str]]): List of item-level properties present in the annotation file
        team_item_properties_lookup (Dict[str, FullProperty]): Server- side state of item-level properties
        client (Client): Darwin Client object
        dataset (RemoteDataset): RemoteDataset object
        console (Console): Rich Console
    """
    if item_properties:
        item_properties_set = {prop["name"] for prop in item_properties}
        for item_property in item_properties_set:
            for team_prop in team_item_properties_lookup:
                if team_prop == item_property:
                    prop_datasets = (
                        team_item_properties_lookup[team_prop].dataset_ids or []
                    )
                    if dataset.dataset_id not in prop_datasets:
                        updated_property = team_item_properties_lookup[team_prop]
                        updated_property.dataset_ids.append(dataset.dataset_id)
                        updated_property.property_values = (
                            []
                        )  # Necessary to clear, otherwise we're trying to add the exsting values to themselves
                        console.print(
                            f"Adding item-level property '{updated_property.name}' to the dataset '{dataset.name}' ",
                            style="info",
                        )
                        client.update_property(dataset.team, updated_property)


def import_annotations(  # noqa: C901
    dataset: "RemoteDataset",
    importer: Callable[[Path], Union[List[dt.AnnotationFile], dt.AnnotationFile, None]],
    file_paths: List[PathLike],
    append: bool,
    class_prompt: bool = True,
    delete_for_empty: bool = False,
    import_annotators: bool = False,
    import_reviewers: bool = False,
    overwrite: bool = False,
    use_multi_cpu: bool = False,
    cpu_limit: Optional[int] = None,
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
    overwrite : bool, default: False
        If ``True`` it will bypass a warning that the import will overwrite the current annotations if any are present.
        If ``False`` this warning will be skipped and the import will overwrite the current annotations without warning.
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

    classes_in_dataset: dt.DictFreeForm = _build_main_annotations_lookup_table(
        [
            cls
            for cls in team_classes
            if cls["available"] or cls["name"] in GLOBAL_CLASSES
        ]
    )
    classes_in_team: dt.DictFreeForm = _build_main_annotations_lookup_table(
        [
            cls
            for cls in team_classes
            if not cls["available"] and cls["name"] not in GLOBAL_CLASSES
        ]
    )
    attributes = _build_attribute_lookup(dataset)

    console.print("Retrieving local annotations ...", style="info")
    local_files = []
    local_files_missing_remotely = []

    remote_files_targeted_by_import = _get_remote_files_targeted_by_import(
        importer, file_paths, dataset, console, use_multi_cpu, cpu_limit
    )
    remote_file_medical_metadata = _get_remote_file_medical_metadata(
        remote_files_targeted_by_import
    )

    if importer.__module__ == "darwin.importer.formats.nifti":
        maybe_parsed_files: Optional[Iterable[dt.AnnotationFile]] = _find_and_parse(
            importer,
            file_paths,
            console,
            use_multi_cpu,
            cpu_limit,
            remote_file_medical_metadata,
        )
    else:
        maybe_parsed_files: Optional[Iterable[dt.AnnotationFile]] = _find_and_parse(
            importer,
            file_paths,
            console,
            use_multi_cpu,
            cpu_limit,
        )

    if not maybe_parsed_files:
        raise ValueError("Not able to parse any files.")

    parsed_files: List[AnnotationFile] = flatten_list(list(maybe_parsed_files))
    parsed_files = _apply_axial_flips_to_annotations(
        parsed_files, remote_file_medical_metadata
    )

    filenames: List[str] = [
        parsed_file.filename for parsed_file in parsed_files if parsed_file is not None
    ]

    console.print("Fetching remote file list...", style="info")
    # This call will only filter by filename; so can return a superset of matched files across different paths
    # There is logic in this function to then include paths to narrow down to the single correct matching file
    remote_files: Dict[str, Dict[str, Any]] = {}

    # Try to fetch files in large chunks; in case the filenames are too large and exceed the url size
    # retry in smaller chunks
    chunk_size = 100
    while chunk_size > 0:
        try:
            remote_files = _get_remote_files_ready_for_import(
                dataset,
                filenames,
                chunk_size,
            )
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

    annotation_format = _get_annotation_format(importer)
    local_files, slot_errors, slot_warnings = _verify_slot_annotation_alignment(
        local_files, remote_files
    )

    _display_slot_warnings_and_errors(
        slot_errors, slot_warnings, annotation_format, console
    )

    if annotation_format == "darwin":
        dataset.client.load_feature_flags()

        # Check if the flag exists. When the flag is deprecated in the future we will always perform this check
        static_instance_id_feature_flag_exists = any(
            feature.name == "STATIC_INSTANCE_ID"
            for feature in dataset.client.features.get(dataset.team, [])
        )
        check_for_multi_instance_id_annotations = (
            static_instance_id_feature_flag_exists
            and dataset.client.feature_enabled("STATIC_INSTANCE_ID")
        ) or not static_instance_id_feature_flag_exists

        if check_for_multi_instance_id_annotations:
            _warn_for_annotations_with_multiple_instance_ids(local_files, console)

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

        remote_classes = _build_main_annotations_lookup_table(maybe_remote_classes)
    else:
        remote_classes = _build_main_annotations_lookup_table(team_classes)

    if delete_for_empty:
        console.print(
            "Importing annotations...\nEmpty annotation file(s) will clear all existing annotations in matching remote files.",
            style="info",
        )
    else:
        console.print(
            "Importing annotations...\nEmpty annotations will be skipped, if you want to delete annotations rerun with '--delete-for-empty'.",
            style="info",
        )

    if not append and not overwrite:
        continue_to_overwrite = _overwrite_warning(
            dataset.client, dataset, local_files, remote_files, console
        )
        if not continue_to_overwrite:
            return

    def import_annotation(parsed_file):
        image_id = remote_files[parsed_file.full_path]["item_id"]
        default_slot_name = remote_files[parsed_file.full_path]["slot_names"][0]
        if parsed_file.slots and parsed_file.slots[0].name:
            default_slot_name = parsed_file.slots[0].name

        metadata_path = is_properties_enabled(parsed_file.path)

        errors, _ = _import_annotations(
            dataset.client,
            image_id,
            remote_classes,
            attributes,
            parsed_file.annotations,
            parsed_file.item_properties,
            default_slot_name,
            dataset,
            append,
            delete_for_empty,
            import_annotators,
            import_reviewers,
            metadata_path,
        )

        if errors:
            console.print(f"Errors importing {parsed_file.filename}", style="error")
            for error in errors:
                console.print(f"\t{error}", style="error")

    def process_local_file(local_file):
        if local_file is None:
            parsed_files = []
        elif not isinstance(local_file, List):
            parsed_files = [local_file]
        else:
            parsed_files = local_file

        # Remove files missing on the server
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
            if not (file_to_track.annotations or file_to_track.item_properties)
            and (not delete_for_empty)
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

            if use_multi_cpu:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=cpu_limit
                ) as executor:
                    futures = [
                        executor.submit(import_annotation, file)
                        for file in files_to_track
                    ]
                    for _ in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="Importing annotations from local file",
                    ):
                        future = next(concurrent.futures.as_completed(futures))
                        try:
                            future.result()
                        except Exception as exc:
                            console.print(
                                f"Generated an exception: {exc}", style="error"
                            )
            else:
                for file in tqdm(
                    files_to_track, desc="Importing annotations from local file"
                ):
                    import_annotation(file)

    if use_multi_cpu:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_limit) as executor:
            futures = [
                executor.submit(process_local_file, local_file)
                for local_file in local_files
            ]
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing local annotation files",
            ):
                future = next(concurrent.futures.as_completed(futures))
                try:
                    future.result()
                except Exception as exc:
                    console.print(f"Generated an exception: {exc}", style="error")
    else:
        for local_file in tqdm(local_files, desc="Processing local annotation files"):
            process_local_file(local_file)


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
    include_empty_attributes: Optional[bool] = False,
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

    if not data.get("attributes") and include_empty_attributes:
        data["attributes"] = {"attributes": []}

    return data


def _format_polygon_for_import(
    annotation: dt.Annotation, data: dt.DictFreeForm
) -> dt.DictFreeForm:
    if "polygon" in data:
        if len(annotation.data["paths"]) > 1:
            data["polygon"] = {
                "path": annotation.data["paths"][0],
                "additional_paths": annotation.data["paths"][1:],
            }
        elif len(annotation.data["paths"]) == 1:
            data["polygon"] = {"path": annotation.data["paths"][0]}
    return data


def _annotators_or_reviewers_to_payload(
    actors: List[dt.AnnotationAuthor], role: dt.AnnotationAuthorRole
) -> List[dt.DictFreeForm]:
    return [{"email": actor.email, "role": role.value} for actor in actors]


def _handle_reviewers(
    import_reviewers: bool,
    annotation: Optional[dt.Annotation] = None,
    item_property_value: Optional[Dict[str, Any]] = None,
) -> List[dt.DictFreeForm]:
    if import_reviewers:
        if annotation and annotation.reviewers:
            return _annotators_or_reviewers_to_payload(
                annotation.reviewers, dt.AnnotationAuthorRole.REVIEWER
            )
        elif item_property_value and "reviewers" in item_property_value:
            return _annotators_or_reviewers_to_payload(
                _parse_annotators(item_property_value["reviewers"]),
                dt.AnnotationAuthorRole.REVIEWER,
            )
    return []


def _handle_annotators(
    import_annotators: bool,
    annotation: Optional[dt.Annotation] = None,
    item_property_value: Optional[Dict[str, Any]] = None,
) -> List[dt.DictFreeForm]:
    if import_annotators:
        if annotation and annotation.annotators:
            return _annotators_or_reviewers_to_payload(
                annotation.annotators, dt.AnnotationAuthorRole.ANNOTATOR
            )
        elif item_property_value and "annotators" in item_property_value:
            return _annotators_or_reviewers_to_payload(
                _parse_annotators(item_property_value["annotators"]),
                dt.AnnotationAuthorRole.ANNOTATOR,
            )
    return []


def _handle_video_annotation_subs(annotation: dt.VideoAnnotation):
    """
    Remove duplicate sub-annotations from the VideoAnnotation.annotation(s) to be imported.
    """
    last_subs = None
    for frame_index, _annotation in annotation.frames.items():
        _annotation: dt.Annotation
        subs = []
        for sub in _annotation.subs:
            if (
                last_subs is not None
                and all(
                    any(
                        last_sub.annotation_type == sub.annotation_type
                        and last_sub.data == sub.data
                        for last_sub in last_subs
                    )
                    for sub in _annotation.subs
                )
                and not annotation.keyframes[frame_index]
            ):
                # drop sub-annotation whenever we know it didn't change since last one
                # which likely wouldn't create on backend side sub-annotation keyframe.
                # this is a workaround for the backend not handling duplicate sub-annotations.
                continue
            subs.append(sub)
        last_subs = _annotation.subs
        _annotation.subs = subs


def _get_annotation_data(
    annotation: dt.AnnotationLike, annotation_class_id: str, attributes: dt.DictFreeForm
) -> dt.DictFreeForm:
    annotation_class = annotation.annotation_class
    if isinstance(annotation, dt.VideoAnnotation):
        _handle_video_annotation_subs(annotation)
        data = annotation.get_data(
            only_keyframes=True,
            post_processing=lambda annotation, data: _handle_subs(
                annotation,
                _format_polygon_for_import(annotation, data),
                annotation_class_id,
                attributes,
                include_empty_attributes=True,
            ),
        )
    else:
        data = {annotation_class.annotation_type: annotation.data}
        data = _format_polygon_for_import(annotation, data)
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


def _parse_empty_masks(
    annotation: dt.Annotation,
    raster_layer: dt.Annotation,
    raster_layer_dense_rle_ids: Optional[Set[str]] = None,
    raster_layer_dense_rle_ids_frames: Optional[Dict[int, Set[str]]] = None,
):
    """
    Check if the mask is empty (i.e. masks that do not have a corresponding raster layer) if so, skip import of the mask.
    This function is used for both dt.Annotation and dt.VideoAnnotation objects.

    Args:
        annotation (dt.Annotation or dt.VideoAnnotation): annotation object to be imported
        raster_layer (dt.Annotation or dt.VideoAnnotation): raster layer object to be imported
        raster_layer_dense_rle_ids (Optional[Set[str]], optional): raster-layer dense_rle_ids. Defaults to None.
        raster_layer_dense_rle_ids_frames (Optional[Dict[int, Set[str]]], optional): raster-layer dense_rle_ids for each frame. Defaults to None.

    Returns:
        tuple[Optional[Set[str]], Optional[Dict[int, Set[str]]]]: raster_layer_dense_rle_ids, raster_layer_dense_rle_ids_frames
    """
    # For dt.VideoAnnotation, create dense_rle ids for each frame.
    if raster_layer_dense_rle_ids_frames is None and isinstance(
        annotation, dt.VideoAnnotation
    ):
        assert isinstance(raster_layer, dt.VideoAnnotation)

        # build a dict of frame_index: set of dense_rle_ids (for each frame in VideoAnnotation object)
        raster_layer_dense_rle_ids_frames = {}
        for frame_index, _rl in raster_layer.frames.items():
            raster_layer_dense_rle_ids_frames[frame_index] = set(
                _rl.data["dense_rle"][::2]
            )

        # check every frame
        # - if the 'annotation_class_id' is in raster_layer's mask_annotation_ids_mapping dict
        # - if the 'dense_rle_id' is in raster_layer's dense_rle_ids_frames dict
        # if not, skip import of the mask, and remove it from mask_annotation_ids_mapping
        for frame_index, _annotation in annotation.frames.items():
            _annotation_id = _annotation.id
            if (
                frame_index in raster_layer_dense_rle_ids_frames
                and raster_layer.frames[frame_index].data[
                    "mask_annotation_ids_mapping"
                ][_annotation_id]
                not in raster_layer_dense_rle_ids_frames[frame_index]
            ):
                # skip import of the mask, and remove it from mask_annotation_ids_mapping
                logger.warning(
                    f"Skipping import of mask annotation '{_annotation.annotation_class.name}' as it does not have a corresponding raster layer"
                )
                del raster_layer.frames[frame_index]["mask_annotation_ids_mapping"][
                    _annotation_id
                ]
                return raster_layer_dense_rle_ids, raster_layer_dense_rle_ids_frames

    # For dt.Annotation, create dense_rle ids.
    elif raster_layer_dense_rle_ids is None and isinstance(annotation, dt.Annotation):
        assert isinstance(raster_layer, dt.Annotation)

        # build a set of dense_rle_ids (for the Annotation object)
        raster_layer_dense_rle_ids = set(raster_layer.data["dense_rle"][::2])

        # check the annotation (i.e. mask)
        # - if the 'annotation_class_id' is in raster_layer's mask_annotation_ids_mapping dict
        # - if the 'dense_rle_id' is in raster_layer's dense_rle_ids dict
        # if not, skip import of the mask, and remove it from mask_annotation_ids_mapping
        _annotation_id = annotation.id
        if (
            raster_layer.data["mask_annotation_ids_mapping"][_annotation_id]
            not in raster_layer_dense_rle_ids
        ):
            # skip import of the mask, and remove it from mask_annotation_ids_mapping
            logger.warning(
                f"Skipping import of mask annotation '{annotation.annotation_class.name}' as it does not have a corresponding raster layer"
            )
            del raster_layer.data["mask_annotation_ids_mapping"][_annotation_id]
            return raster_layer_dense_rle_ids, raster_layer_dense_rle_ids_frames

    return raster_layer_dense_rle_ids, raster_layer_dense_rle_ids_frames


def _import_annotations(
    client: "Client",  # TODO: This is unused, should it be?
    id: Union[str, int],
    remote_classes: dt.DictFreeForm,
    attributes: dt.DictFreeForm,
    annotations: List[dt.Annotation],
    item_properties: List[Dict[str, str]],
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

    raster_layer: Optional[dt.Annotation] = None
    raster_layer_dense_rle_ids: Optional[Set[str]] = None
    raster_layer_dense_rle_ids_frames: Optional[Dict[int, Set[str]]] = None
    serialized_annotations = []
    annotation_class_ids_map: Dict[Tuple[str, str], str] = {}
    for annotation in annotations:
        annotation_class = annotation.annotation_class
        annotation_type = (
            annotation_class.annotation_internal_type
            or annotation_class.annotation_type
        )

        if (
            (
                annotation_type not in remote_classes
                or annotation_class.name not in remote_classes[annotation_type]
            )
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
            if raster_layer is None:
                raster_layer = next(
                    (
                        a
                        for a in annotations
                        if a.annotation_class.annotation_type == "raster_layer"
                    ),
                    None,
                )
            if raster_layer:
                (
                    raster_layer_dense_rle_ids,
                    raster_layer_dense_rle_ids_frames,
                ) = _parse_empty_masks(
                    annotation,
                    raster_layer,
                    raster_layer_dense_rle_ids,
                    raster_layer_dense_rle_ids_frames,
                )

        actors: List[dt.DictFreeForm] = []
        actors.extend(_handle_annotators(import_annotators, annotation=annotation))
        actors.extend(_handle_reviewers(import_reviewers, annotation=annotation))

        # Insert the default slot name if not available in the import source
        annotation = _handle_slot_names(annotation, dataset.version, default_slot_name)

        annotation_class_ids_map[(annotation_class.name, annotation_type)] = (
            annotation_class_id
        )
        serial_obj = {
            "annotation_class_id": annotation_class_id,
            "data": data,
            "context_keys": {"slot_names": annotation.slot_names},
        }

        annotation.id = annotation.id or str(uuid.uuid4())
        serial_obj["id"] = annotation.id

        if actors:
            serial_obj["actors"] = actors  # type: ignore

        serialized_annotations.append(serial_obj)

    annotation_id_property_map = _import_properties(
        metadata_path,
        item_properties,
        client,
        annotations,  # type: ignore
        annotation_class_ids_map,
        dataset,
    )

    _update_payload_with_properties(serialized_annotations, annotation_id_property_map)
    serialized_item_level_properties = _serialize_item_level_properties(
        item_properties, client, dataset, import_annotators, import_reviewers
    )

    payload: dt.DictFreeForm = {"annotations": serialized_annotations}
    if serialized_item_level_properties:
        payload["properties"] = serialized_item_level_properties
    payload["overwrite"] = _get_overwrite_value(append)

    try:
        dataset.import_annotation(id, payload=payload)
    except RequestEntitySizeExceeded:
        logger.warning(
            "Annotation payload exceeds request entity size. Splitting payload into smaller chunks for import."
        )
        payloads = _split_payloads(payload)
        for chunked_payload in payloads:
            try:
                dataset.import_annotation(id, payload=chunked_payload)
            except Exception as e:
                errors.append(e)
                success = dt.Success.FAILURE
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


def _overwrite_warning(
    client: "Client",
    dataset: "RemoteDataset",
    local_files: List[dt.AnnotationFile],
    remote_files: Dict[str, Dict[str, Any]],
    console: Console,
) -> bool:
    """
    Determines if any dataset items targeted for import already have annotations or item-level properties that will be overwritten.
    If they do, a warning is displayed to the user and they are prompted to confirm if they want to proceed with the import.

    Parameters
    ----------
    client : Client
        The Darwin Client object.
    dataset : RemoteDataset
        The dataset where the annotations will be imported.
    files : List[dt.AnnotationFile]
        The list of local annotation files to will be imported.
    remote_files : Dict[str, Tuple[str, str]]
        A dictionary of the remote files in the dataset.
    console : Console
        The console object.

    Returns
    -------
    bool
        True if the user wants to proceed with the import, False otherwise.
    """
    files_with_annotations_to_overwrite = []
    files_with_item_properties_to_overwrite = []

    for local_file in local_files:
        item_id = remote_files.get(local_file.full_path)["item_id"]  # type: ignore

        # Check if the item has annotations that will be overwritten
        remote_annotations = client.api_v2._get_remote_annotations(
            item_id,
            dataset.team,
        )
        if (
            remote_annotations
            and local_file.full_path not in files_with_annotations_to_overwrite
        ):
            files_with_annotations_to_overwrite.append(local_file.full_path)

        # Check if the item has item-level properties that will be overwritten
        if local_file.item_properties:
            response: Dict[str, List[Dict[str, str]]] = (
                client.api_v2._get_properties_state_for_item(item_id, dataset.team)
            )
            item_property_ids_with_populated_values = [
                property_data["id"]
                for property_data in response["properties"]
                if property_data["values"]
            ]
            if item_property_ids_with_populated_values:
                files_with_item_properties_to_overwrite.append(local_file.full_path)

    if files_with_annotations_to_overwrite or files_with_item_properties_to_overwrite:
        # Overwriting of annotations
        if files_with_annotations_to_overwrite:
            console.print(
                f"The following {len(files_with_annotations_to_overwrite)} dataset item(s) have annotations that will be overwritten by this import:",
                style="warning",
            )
            for file in files_with_annotations_to_overwrite:
                console.print(f"- {file}", style="warning")

        # Overwriting of item-level-properties
        if files_with_item_properties_to_overwrite:
            console.print(
                f"The following {len(files_with_item_properties_to_overwrite)} dataset item(s) have item-level properties that will be overwritten by this import:",
                style="warning",
            )
            for file in files_with_item_properties_to_overwrite:
                console.print(f"- {file}", style="warning")

        proceed = input("Do you want to proceed with the import? [y/N] ")
        if proceed.lower() != "y":
            return False
    return True


def _get_annotation_format(
    importer: Callable[[Path], Union[List[dt.AnnotationFile], dt.AnnotationFile, None]],
) -> str:
    """
    Returns the annotation format of the importer used to parse local annotation files

    Parameters
    ----------
    importer : Callable[[Path], Union[List[dt.AnnotationFile], dt.AnnotationFile, None]]
        The importer used to parse local annotation files

    Returns
    -------
    annotation_format : str
        The annotation format of the importer used to parse local files
    """
    return importer.__module__.split(".")[3]


def _verify_slot_annotation_alignment(
    local_files: List[dt.AnnotationFile],
    remote_files: Dict[str, Dict[str, Any]],
) -> Tuple[List[dt.AnnotationFile], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Runs slot alignment validation against annotations being imported. The following checks are run:
    - For multi-slotted items:
        - For every annotation not uploaded to a specific slot:
          A non-blocking warning is generated explaining that it will be uploaded to the default slot

    - For multi-channeled items:
        - For every annotation not uploaded to a specific slot:
          A non-blocking warning is generated explaining that it will be uploaded to the base slot
        - For every annotation uploaded to a slot other than the base slot:
          A blocking error is generated explaining that annotations can only be uploaded to the base slot of multi-channeled items

    Files that generate exclusively non-blocking warnings will have those warnings displayed, but their import will continue.
    Files that generate any blocking error will only have their blocking errors displayed, and they are removed from the list of files to be imported.

    Errors can only be generated by referring to multi-slotted or multi-channeled items.
    These concepts are only supported by Darwin JSON 2.0, so stop imports of other formats if any warnings occur.

    Parameters
    ----------
    local_files : List[dt.AnnotationFile]
        A list of local annotation files to be uploaded
    remote_files : Dict[str, Dict[str, Any]]
        Information about each remote dataset item that corresponds to the local annotation file being uploaded

    Returns
    -------
    local_files : List[dt.AnnotationFile]
        A pruned list of the input annotation flies. It excludes any input files that generated a blocking warning
    slot_errors : Dict[str, List[str]]
        A dictionary of blocking errors for each file
    slot_warnings : Dict[str, List[str]]
        A dictionary of non-blocking warnings for each file
    """

    slot_errors, slot_warnings = {}, {}
    for local_file in local_files:
        remote_file = remote_files[local_file.full_path]
        local_file_path = str(local_file.path)
        if len(remote_file["slot_names"]) == 1:
            continue  # Skip single-slotted items
        base_slot = remote_file["slot_names"][0]
        layout_version = remote_file["layout"]["version"]
        if layout_version == 1 or layout_version == 2:  # Multi-slotted item
            for annotation in local_file.annotations:
                try:
                    annotation_slot = annotation.slot_names[0]
                except IndexError:
                    if local_file_path not in slot_warnings:
                        slot_warnings[local_file_path] = []
                    slot_warnings[local_file_path].append(
                        f"Annotation imported to multi-slotted item not assigned slot. Uploading to the default slot: {base_slot}"
                    )

        elif layout_version == 3:  # Channeled item
            for annotation in local_file.annotations:
                try:
                    annotation_slot = annotation.slot_names[0]
                except IndexError:
                    if local_file_path not in slot_warnings:
                        slot_warnings[local_file_path] = []
                    slot_warnings[local_file_path].append(
                        f"Annotation imported to multi-channeled item not assigned a slot. Uploading to the base slot: {base_slot}"
                    )
                    annotation.slot_names = [base_slot]
                    continue
                if annotation_slot != base_slot:
                    if local_file_path not in slot_errors:
                        slot_errors[local_file_path] = []
                    slot_errors[local_file_path].append(
                        f"Annotation is linked to slot {annotation_slot} of the multi-channeled item {local_file.full_path}. Annotations uploaded to multi-channeled items have to be uploaded to the base slot, which for this item is {base_slot}."
                    )
        else:
            raise Exception(f"Unknown layout version: {layout_version}")

    # Remove non-blocking warnings if there are corresponding blocking warnings
    for key in slot_errors.keys():
        if key in slot_warnings:
            del slot_warnings[key]

    local_files = [
        local_file
        for local_file in local_files
        if str(local_file.path) not in slot_errors
    ]

    return local_files, slot_errors, slot_warnings


def _display_slot_warnings_and_errors(
    slot_errors: Dict[str, List[str]],
    slot_warnings: Dict[str, List[str]],
    annotation_format: str,
    console: Console,
) -> None:
    """
    Displays slot warnings and errors.

    Parameters
    ----------
    local_files : List[dt.AnnotationFile]
        A list of local annotation files to be uploaded
    slot_errors : Dict[str, List[str]]
        A dictionary of blocking warnings for each file
    slot_warnings : Dict[str, List[str]]
        A dictionary of non-blocking warnings for each file
    annotation_format : str
        The annotation format of the importer used to parse local files
    console : Console
        The console object

    Raises
    ------
    TypeError
        If there are any warnings generated and the annotation format is not Darwin JSON 2.0 or NifTI
    """

    # Warnings can only be generated by referring to slots, which is only supported by the Darwin JSON & NiFTI formats
    # Therefore, stop imports of all other formats if there are any warnings
    supported_formats = ["darwin", "nifti"]
    if (slot_errors or slot_warnings) and annotation_format not in supported_formats:
        raise TypeError(
            "You are attempting to import annotations to multi-slotted or multi-channeled items using an annotation format that doesn't support them. To import annotations to multi-slotted or multi-channeled items, please use the Darwin JSON 2.0 format: https://docs.v7labs.com/reference/darwin-json"
        )
    if slot_warnings:
        console.print(
            f"WARNING: {len(slot_warnings)} file(s) have the following non-blocking warnings. Imports of these files will continue:",
            style="warning",
        )
        for file in slot_warnings:
            console.print(f"- File: {file}, warnings:", style="info")
            for warning in slot_warnings[file]:
                console.print(f"  - {warning}")

    if slot_errors:
        console.print(
            f"WARNING: {len(slot_errors)} file(s) have the following blocking issues and will not be imported. Please resolve these issues and re-import them.",
            style="warning",
        )
        for file in slot_errors:
            console.print(f"- File: {file}, issues:", style="info")
            for warning in slot_errors[file]:
                console.print(f"  - {warning}")


def _warn_for_annotations_with_multiple_instance_ids(
    local_files: List[dt.AnnotationFile], console: Console
) -> None:
    """
    Warns the user if any video annotations have multiple unique instance IDs.

    This function checks each video annotation in the provided list of local annotation
    files for multiple instance ID values. A warning is printed to the console for each
    instance of this occurrence.

    Parameters
    ----------
    local_files : List[dt.AnnotationFile]
        A list of local annotation files to be checked.
    console : Console
        The console object used to print warnings and messages.
    """
    files_with_multi_instance_id_annotations = {}
    files_with_video_annotations = [
        local_file for local_file in local_files if local_file.is_video
    ]
    for file in files_with_video_annotations:
        for annotation in file.annotations:
            unique_instance_ids = []
            for frame_idx in annotation.frames:  # type: ignore
                for subannotation in annotation.frames[frame_idx].subs:  # type: ignore
                    if subannotation.annotation_type == "instance_id":
                        instance_id = subannotation.data
                        if instance_id not in unique_instance_ids:
                            unique_instance_ids.append(instance_id)

            if len(unique_instance_ids) > 1:
                if file.path not in files_with_multi_instance_id_annotations:
                    files_with_multi_instance_id_annotations[file.path] = 1
                else:
                    files_with_multi_instance_id_annotations[file.path] += 1

    if files_with_multi_instance_id_annotations:
        console.print(
            "The following files have annotation(s) with multiple instance ID values. Instance IDs are static, so only the first instance ID of each annotation will be imported:",
            style="warning",
        )
        for file in files_with_multi_instance_id_annotations:
            console.print(
                f"- File: {file} has {files_with_multi_instance_id_annotations[file]} annotation(s) with multiple instance IDs"
            )


def _split_payloads(
    payload: Dict[str, Any], max_payload_size: int = 32_000_000
) -> List[Dict[str, Any]]:
    """
    This function takes an input payload and splits it into smaller payloads, ensuring each chunk does not exceed the specified maximum size.
    This is useful when importing annotations, as it prevents HTTP 413 errors (`RequestEntitySizeExceeded`) from occurring due to large request entity sizes.

    Parameters
    ----------
    payload : Dict[str, Any]
        The input payload to be split.
    max_payload_size : int, optional
        The maximum size of each split payload. Defaults to 32,000,000 bytes.

    Returns
    -------
    List[Dict[str, Any]]
        A list of split payloads, each not exceeding the specified maximum size.

    Raises
    ------
    ValueError
        If any single annotation exceeds the `max_payload_size` limit
    """
    payloads = []
    base_payload = {"annotations": [], "overwrite": payload["overwrite"]}
    current_payload = copy.deepcopy(base_payload)
    current_payload_size = 0

    for annotation in payload["annotations"]:
        annotation_size = len(json.dumps({"annotations": [annotation]}).encode("utf-8"))
        if current_payload_size + annotation_size < max_payload_size:
            current_payload["annotations"].append(annotation)
            current_payload_size += annotation_size
        else:
            if annotation_size > max_payload_size:
                raise ValueError(
                    f"One or more annotations exceed the maximum allowed size of 32 MiB ({max_payload_size})"
                )
            payloads.append(current_payload)
            current_payload = copy.deepcopy(base_payload)
            current_payload["overwrite"] = (
                False  # Required to make sure subsequent payloads don't overwrite previous ones
            )
            current_payload["annotations"].append(annotation)
            current_payload_size = annotation_size

    if current_payload["annotations"]:
        payloads.append(current_payload)

    return payloads


def _get_remote_files_targeted_by_import(
    importer: Callable[[Path], Union[List[dt.AnnotationFile], dt.AnnotationFile, None]],
    file_paths: List[PathLike],
    dataset: "RemoteDataset",
    console: Optional[Console] = None,
    use_multi_cpu: bool = True,
    cpu_limit: int = 1,
) -> List[DatasetItem]:
    """
    Parses local annotations files for import and returns a list of remote dataset items
    targeted by the import. Handles chunking of requests if there are many files to
    avoid URL length issues.
    Parameters
    ----------
    importer: Callable[[Path], Union[List[dt.AnnotationFile], dt.AnnotationFile, None]]
        The importer used to parse local annotation files
    file_paths: List[PathLike]
        A list of local annotation files to be uploaded
    dataset: RemoteDataset
        The remote dataset to fetch files from
    console: Optional[Console]
        The console object
    use_multi_cpu: bool
        Whether to use multi-CPU processing
    cpu_limit: int
        The number of CPUs to use for processing
    Returns
    -------
    List[DatasetItem]
        A list of remote dataset items targeted by the import
    Raises
    ------
    ValueError
        If no files could be parsed or if the URL becomes too long even with minimum chunk size
    """
    maybe_parsed_files = _find_and_parse(
        importer,
        file_paths,
        console,
        use_multi_cpu,
        cpu_limit,
    )
    if not maybe_parsed_files:
        raise ValueError("Not able to parse any files.")

    remote_filenames = list({file.filename for file in maybe_parsed_files})
    remote_filepaths = [file.full_path for file in maybe_parsed_files]

    all_remote_files: List[DatasetItem] = []
    current_chunk: List[str] = []
    current_length = BASE_URL_LENGTH
    max_chunk_length = MAX_URL_LENGTH - BASE_URL_LENGTH

    for filename in remote_filenames:
        filename_length = len(filename) + FILENAME_OVERHEAD
        if current_length + filename_length > max_chunk_length and current_chunk:
            remote_files = dataset.fetch_remote_files(
                filters={"item_names": current_chunk}
            )
            all_remote_files.extend(remote_files)
            current_chunk = []
            current_length = BASE_URL_LENGTH

        current_chunk.append(filename)
        current_length += filename_length

    if current_chunk:
        remote_files = dataset.fetch_remote_files(filters={"item_names": current_chunk})
        all_remote_files.extend(remote_files)

    return [
        remote_file
        for remote_file in all_remote_files
        if remote_file.full_path in remote_filepaths
    ]


def _get_remote_file_medical_metadata(
    remote_files_targeted_by_import: List[DatasetItem],
) -> Dict[Path, Dict[str, Any]]:
    """
    This function parses the remote files targeted by the import and returns metadata
    for medical files. For each medical file, it returns a dictionary containing:
    - For each slot:
        - legacy: Whether the slot requires legacy NifTI scaling
        - affine: The affine matrix (if legacy)
        - original_affine: The original affine matrix (if legacy)
        - axial_flips: A list of 3 values that are either 1 or -1. -1 indicates the axis is reversed.
        - pixdims: The (x, y, z) pixel dimensions
        - height: The height of the file
        - width: The width of the file
        - num_frames: The number of frames in the file (only present for sequence files)
        - primary_plane: The primary plane of the medical image

    Parameters
    ----------
    remote_files_targeted_by_import: List[DatasetItem]
        The remote files targeted by the import

    Returns
    -------
    Dict[Path, Dict[str, Any]]
        A dictionary mapping file paths to their medical metadata:
        {
            remote_file_path: {
                slot_name: {
                    legacy: bool,
                    affine: ndarray,
                    original_affine: ndarray,
                    axial_flips: ndarray,
                    pixdims: List[float],
                    height: int,
                    width: int,
                    num_frames: int, (Only present for sequence files)
                    primary_plane: str,
                }
            }
        }
    """
    remote_file_medical_metadata = {}
    for remote_file in remote_files_targeted_by_import:
        if not remote_file.slots:
            continue

        file_metadata = {}
        for slot in remote_file.slots:
            if not slot_is_medical(slot):
                continue

            slot_name = slot["slot_name"]
            legacy = not slot_is_handled_by_monai(slot)
            affine_str = slot["metadata"]["medical"]["affine"]
            original_affine_str = slot["metadata"]["medical"]["original_affine"]
            affine = np.array(affine_str).astype(float)
            original_affine = np.array(original_affine_str).astype(float)
            pixdims = slot["metadata"]["medical"]["pixdims"]
            width = slot["metadata"]["width"]
            height = slot["metadata"]["height"]
            slot_metadata = {
                "legacy": legacy,
                "affine": affine,
                "original_affine": original_affine,
                "pixdims": pixdims,
                "width": width,
                "height": height,
            }
            num_frames = slot.get("total_sections", None)
            if num_frames:
                slot_metadata["num_frames"] = num_frames

            if legacy:
                axial_flips = [1, 1, 1]
                primary_plane = "AXIAL"
            else:
                ax_codes = nib.orientations.aff2axcodes(affine)
                original_ax_codes = nib.orientations.aff2axcodes(original_affine)
                ornt = nib.orientations.axcodes2ornt(ax_codes)
                original_ornt = nib.orientations.axcodes2ornt(original_ax_codes)
                axial_flips = nib.orientations.ornt_transform(original_ornt, ornt)[:, 1]
                primary_plane = slot["metadata"]["medical"]["plane_map"][slot_name]

            slot_metadata.update(
                {"axial_flips": axial_flips, "primary_plane": primary_plane}
            )
            file_metadata[slot_name] = slot_metadata

        if file_metadata:
            file_path = Path(remote_file.full_path)
            remote_file_medical_metadata[file_path] = file_metadata

    return remote_file_medical_metadata


def slot_is_medical(slot: Dict[str, Any]) -> bool:
    return slot.get("metadata", {}).get("medical") is not None


def slot_is_handled_by_monai(slot: Dict[str, Any]) -> bool:
    return slot.get("metadata", {}).get("medical", {}).get("handler") == "MONAI"


def _apply_axial_flips_to_annotations(
    parsed_files: List[dt.AnnotationFile],
    remote_file_medical_metadata: Dict[Path, Dict[str, Any]],
) -> List[dt.AnnotationFile]:
    for parsed_file in parsed_files:
        remote_file_path = Path(parsed_file.full_path)

        if remote_file_path in remote_file_medical_metadata:
            medical_metadata = remote_file_medical_metadata[remote_file_path]

            for annotation in parsed_file.annotations:
                annotation_slot_name = annotation.slot_names[0]
                if annotation_slot_name not in medical_metadata:
                    continue

                slot_metadata = medical_metadata[annotation_slot_name]
                if slot_metadata["legacy"]:
                    continue

                axial_flips = slot_metadata["axial_flips"]
                if isinstance(annotation, dt.VideoAnnotation):

                    flipped_frames = {}
                    for frame_idx, frame_annotation in annotation.frames.items():
                        flipped_frames[frame_idx] = copy.deepcopy(frame_annotation)

                    if axial_flips[0] == -1:
                        for frame_idx, frame_annotation in flipped_frames.items():
                            if (
                                frame_annotation.annotation_class.annotation_type
                                == "raster_layer"
                            ):
                                frame_annotation._flip_raster_layer_in_x_or_y(
                                    width=slot_metadata["width"],
                                    height=slot_metadata["height"],
                                    axis=dt.CartesianAxis.X,
                                )
                            else:
                                frame_annotation._flip_annotation_in_x_or_y(
                                    axis_limit=slot_metadata["width"],
                                    axis=dt.CartesianAxis.X,
                                )

                    annotation.frames = flipped_frames
                    if axial_flips[1] == -1:
                        for frame_idx, frame_annotation in flipped_frames.items():
                            if (
                                frame_annotation.annotation_class.annotation_type
                                == "raster_layer"
                            ):
                                frame_annotation._flip_raster_layer_in_x_or_y(
                                    width=slot_metadata["width"],
                                    height=slot_metadata["height"],
                                    axis=dt.CartesianAxis.Y,
                                )
                            else:
                                frame_annotation._flip_annotation_in_x_or_y(
                                    axis_limit=slot_metadata["height"],
                                    axis=dt.CartesianAxis.Y,
                                )

                    annotation.frames = flipped_frames
                    if axial_flips[2] == -1:
                        if isinstance(annotation, dt.VideoAnnotation):
                            annotation = _flip_annotation_in_z(
                                annotation=annotation,
                                num_frames=slot_metadata["num_frames"],
                            )
                else:
                    if axial_flips[0] == -1:
                        annotation._flip_annotation_in_x_or_y(
                            axis_limit=slot_metadata["width"],
                            axis=dt.CartesianAxis.X,
                        )
                    if axial_flips[1] == -1:
                        annotation._flip_annotation_in_x_or_y(
                            axis_limit=slot_metadata["height"],
                            axis=dt.CartesianAxis.Y,
                        )

    return parsed_files


def _flip_annotation_in_z(
    annotation: dt.VideoAnnotation,
    num_frames: int,
) -> dt.VideoAnnotation:
    flipped_frames, flipped_keyframes = {}, {}
    for frame_idx, frame_annotation in annotation.frames.items():
        flipped_frame_idx = num_frames - frame_idx - 1
        flipped_frames[flipped_frame_idx] = frame_annotation
    for frame_idx in annotation.keyframes:
        flipped_frame_idx = num_frames - frame_idx - 1
        flipped_keyframes[flipped_frame_idx] = annotation.keyframes[frame_idx]
    segment_start = annotation.segments[0][0]
    segment_end = annotation.segments[0][1]
    flipped_segments = [[num_frames - segment_end, num_frames - segment_start]]
    annotation.frames = flipped_frames
    annotation.keyframes = flipped_keyframes
    annotation.segments = flipped_segments
    # NOTE: We haven't implemented flipping for `hidden_areas` because hidden areas
    # are not supported in slots containing medical data
    return annotation
