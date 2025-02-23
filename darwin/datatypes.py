from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydantic import BaseModel

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = Any  # type:ignore
import numpy as np
from darwin.future.data_objects.properties import (
    PropertyType,
    SelectedProperty,
    PropertyGranularity,
)
from darwin.path_utils import construct_full_path, is_properties_enabled, parse_metadata

# Utility types

NumberLike = Union[
    int, float
]  # Used for functions that can take either an int or a float
# Used for functions that _genuinely_ don't know what type they're dealing with, such as those that test if something is of a certain type.
UnknownType = Any  # type:ignore

# Specific types

ErrorList = List[Exception]


class Success(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    PARTIAL_SUCCESS = auto()
    UNDETERMINED = auto()


Point = Dict[str, float]
BoundingBox = Dict[str, float]
Polygon = List[Point]
ComplexPolygon = List[Polygon]
Node = Dict[str, UnknownType]
EllipseData = Dict[str, Union[float, Point]]
CuboidData = Dict[str, Dict[str, float]]
Segment = List[int]
HiddenArea = List[int]

DarwinVersionNumber = Tuple[int, int, int]

PathLike = Union[str, Path]
ErrorHandler = Callable[[int, str], None]

ItemId = Union[str, int]

# Types that assist in handling JSON payloads
JSONFreeForm = Dict[str, UnknownType]
DictFreeForm = JSONFreeForm
KeyValuePairDict = Dict[str, UnknownType]


class JSONType:
    def __init__(self, **kwargs: JSONFreeForm):
        self.__dict__.update(kwargs)

    def to_json(self) -> JSONFreeForm:
        return self.__dict__

    @classmethod
    def from_json(cls, json: JSONFreeForm) -> "JSONType":
        return cls(**json)

    @classmethod
    def from_dict(cls, json: JSONFreeForm) -> "JSONType":
        return cls(**json)


AnnotationType = Literal[  # NB: Some of these are not supported yet
    "bounding_box",
    "polygon",
    "ellipse",
    "cuboid",
    "segmentation",
    "raster_layer",
    "mask",
    "keypoint",
    "tag",
    "line",
    "skeleton",
    "table",
    "string",
    "graph",
]


@dataclass
class Team:
    """
    Definition of a V7 team.
    """

    #: If this is the default Team or not.
    default: bool

    #: This team's slug.
    slug: str

    #: The path to the directory of all datasets this teams contains.
    datasets_dir: str

    #: The API key used to authenticate for this Team.
    api_key: str

    #: If this is the currently active Team. Defaults to ``False``.
    selected: bool = False


@dataclass(frozen=True)
class Feature:
    """
    Structured payload of a Feature record on V7 Darwin.
    """

    #: The name of this ``Feature``.
    name: str

    #: Whether or not this ``Feature`` is enabled
    #: Disabled ``Feature``\s do nothing, as if they didn't exist.
    enabled: bool


@dataclass(frozen=True, eq=True)
class AnnotationClass:
    """
    Represents an AnnocationClass from an Annotation.
    """

    #:  The name of this ``AnnotationClass``.
    name: str

    #: The type of this ``AnnotationClass``.
    annotation_type: AnnotationType

    #: The V7 internal type of this ``AnnotationClass``.
    #: This is mostly used to convert from types that are known in the outside world by a given
    #: name, but then are known inside V7's lingo by another.
    annotation_internal_type: Optional[str] = None


@dataclass(frozen=True, eq=True)
class SubAnnotation:
    """
    Represents a subannotation that belongs to an AnnotationClass.
    """

    #: The type of this ``SubAnnotation``.
    annotation_type: str

    #: Any external data, in any format, relevant to this ``SubAnnotation``.
    #: Used for compatibility purposes with external formats.
    data: UnknownType


class AnnotationAuthorRole(Enum):
    ANNOTATOR = "annotator"
    REVIEWER = "reviewer"


@dataclass(frozen=True, eq=True)
class AnnotationAuthor:
    """
    Represents an annotation's author
    """

    #: Name of the author
    name: str

    #: Email of the author
    email: str


@dataclass(frozen=False, eq=True)
class Annotation:
    """
    Represents an Annotation from an Image/Video.
    """

    #: The ``AnnotationClass`` from this ``Annotation``.
    annotation_class: AnnotationClass

    #: Any external data, in any format, relevant to this ``Annotation``.
    #: Used for compatibility purposes with external formats.
    data: UnknownType

    #: List of ``SubAnnotations`` belonging to this ``Annotation``.
    subs: List[SubAnnotation] = field(default_factory=list)

    #: V2 slots this annotation belogs to
    slot_names: List[str] = field(default_factory=list)

    #: Authorship of the annotation (annotators)
    annotators: Optional[List[AnnotationAuthor]] = None

    #: Authorship of the annotation (reviewers)
    reviewers: Optional[List[AnnotationAuthor]] = None

    # The darwin ID of this annotation.
    id: Optional[str] = None

    # Properties of this annotation.
    properties: Optional[list[SelectedProperty]] = None

    def get_sub(self, annotation_type: str) -> Optional[SubAnnotation]:
        """
        Returns the first SubAnnotation that matches the given type.

        Parameters
        ----------
        annotation_type: str
            The type of the subannotation.

        Returns
        -------
        Optional[SubAnnotation]
            A SubAnnotation found, or `None` if none was found.
        """
        for sub in self.subs:
            if sub.annotation_type == annotation_type:
                return sub
        return None

    def _flip_annotation_in_x_or_y(self, axis_limit: int, axis: CartesianAxis):
        """
        Flips a coordinate-based annotation in either X or Y axis.

        Parameters
        ----------
        axis_limit : int
            The limit of the axis to flip along.
        """
        annotation_type = (
            self.annotation_class.annotation_type
            if hasattr(self, "annotation_class")
            else None
        )
        if annotation_type == "bounding_box":
            if axis == CartesianAxis.X:
                self.data["x"] = axis_limit - self.data["x"]
            elif axis == CartesianAxis.Y:
                self.data["y"] = axis_limit - self.data["y"]

        elif annotation_type == "polygon":
            if axis == CartesianAxis.X:
                for path in self.data["paths"]:
                    for point in path:
                        point["x"] = axis_limit - point["x"]

            elif axis == CartesianAxis.Y:
                for path in self.data["paths"]:
                    for point in path:
                        point["y"] = axis_limit - point["y"]

        elif annotation_type == "ellipse":
            if axis == CartesianAxis.X:
                self.data["center"]["x"] = axis_limit - self.data["center"]["x"]
            elif axis == CartesianAxis.Y:
                self.data["center"]["y"] = axis_limit - self.data["center"]["y"]

        elif annotation_type == "line":
            if axis == CartesianAxis.X:
                for point in self.data["path"]:
                    point["x"] = axis_limit - point["x"]
            elif axis == CartesianAxis.Y:
                for point in self.data["path"]:
                    point["y"] = axis_limit - point["y"]

        elif annotation_type == "keypoint":
            if axis == CartesianAxis.X:
                self.data["x"] = axis_limit - self.data["x"]
            elif axis == CartesianAxis.Y:
                self.data["y"] = axis_limit - self.data["y"]

        elif annotation_type == "skeleton":
            if axis == CartesianAxis.X:
                for node in self.data["nodes"]:
                    node["x"] = axis_limit - node["x"]
            elif axis == CartesianAxis.Y:
                for node in self.data["nodes"]:
                    node["y"] = axis_limit - node["y"]

    def _flip_raster_layer_in_x_or_y(
        self, width: int, height: int, axis: CartesianAxis
    ):
        """
        Flips a raster layer mask in either X or Y axis.

        Parameters
        ----------
        width : int
            Width of the image/mask
        height : int
            Height of the image/mask
        axis : CartesianAxis
            Axis to flip along (X or Y)

        The method works by:
        1. Decoding RLE to a 2D mask
        2. Flipping the mask along the specified axis
        3. Re-encoding back to RLE format
        """
        dense_rle = self.data["dense_rle"]
        total_pixels = width * height
        mask = np.zeros(total_pixels, dtype=np.uint8)
        idx = 0
        for i in range(0, len(dense_rle), 2):
            value = dense_rle[i]
            length = dense_rle[i + 1]
            mask[idx : idx + length] = value
            idx += length

        mask = mask.reshape(height, width)
        if axis == CartesianAxis.X:
            mask = np.fliplr(mask)
        elif axis == CartesianAxis.Y:
            mask = np.flipud(mask)

        mask = mask.ravel()
        rle = []
        count = 1
        current = mask[0]

        for bit in mask[1:]:
            if bit == current:
                count += 1
            else:
                rle.extend([int(current), count])
                current = bit
                count = 1
        rle.extend([int(current), count])
        self.data["dense_rle"] = rle


@dataclass(frozen=False, eq=True)
class VideoAnnotation:
    """
    Represents an Annotation that belongs to a Video.
    """

    #: The ``AnnotationClass`` from this ``VideoAnnotation``.
    annotation_class: AnnotationClass

    #: A dictionary of frames for this ``VideoAnnotation``.
    frames: Dict[int, UnknownType]

    #: The keyframes for this ``VideoAnnotation``.
    #: Keyframes are a selection of frames from the ``frames`` attribute.
    keyframes: Dict[int, bool]

    #: A list of ``Segment``\'s.
    segments: List[Segment]

    #: Whether this ``VideoAnnotation`` is interpolated or not.
    interpolated: bool

    #: V2 slots this annotation belogs to
    slot_names: List[str] = field(default_factory=list)

    #: Authorship of the annotation (annotators)
    annotators: Optional[List[AnnotationAuthor]] = None

    #: Authorship of the annotation (reviewers)
    reviewers: Optional[List[AnnotationAuthor]] = None

    # The darwin ID of this annotation.
    id: Optional[str] = None

    # Properties of this annotation.
    properties: Optional[list[SelectedProperty]] = None

    #: A list of ``HiddenArea``\'s.
    hidden_areas: List[HiddenArea] = field(default_factory=list)

    def get_data(
        self,
        only_keyframes: bool = True,
        post_processing: Optional[
            Callable[[Annotation, UnknownType], UnknownType]
        ] = None,
    ) -> Dict:
        """
        Return the post-processed frames and the additional information from this
        ``VideoAnnotation`` in a dictionary with the format:

        .. code-block:: python

            {
                "frames": {
                    # Post-Processed Frames here
                },
                "segments": [
                    # Segments here
                ]
                "interpolated": True
            }

        Parameters
        ----------
        only_keyframes: bool, default: True
            Whether or not to return only the keyframes. Defaults to ``True``.
        post_processing: Optional[Callable[[Annotation, Any], Any]], default: None
            If given, it processes each frame through the given ``Callabale`` before adding it to the
            returned dictionary. Defaults to ``None``.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the processed frames, the segments of this ``VideoAnnotation``
            and whether or not it is interpolated.
        """
        if not post_processing:

            def post_processing(
                annotation: Annotation, data: UnknownType
            ) -> UnknownType:
                return data  # type: ignore

        output = {
            "frames": {
                frame: {
                    **post_processing(
                        self.frames[frame],  # type: ignore
                        {self.frames[frame].annotation_class.annotation_type: self.frames[frame].data},  # type: ignore
                    ),
                    **{"keyframe": self.keyframes[frame]},  # type: ignore
                }
                for frame in self.frames
                if not only_keyframes or self.keyframes[frame]
            },
            "segments": self.segments,
            "interpolated": self.interpolated,
            "hidden_areas": self.hidden_areas,
        }

        return output


AnnotationLike = Union[Annotation, VideoAnnotation]


@dataclass
class Slot:
    #: Unique slot name in the item.
    name: Optional[str]

    #: Type of slot, e.g. image or dicom
    type: str

    #: Original upload information for the slot
    source_files: List[SourceFile]

    #: Thumbnail url to the file
    thumbnail_url: Optional[str] = None

    #: Width in pixel
    width: Optional[int] = None

    #: Height in pixels
    height: Optional[int] = None

    #: How many sections (eg. frames) does this slot have
    frame_count: Optional[int] = None

    #: A url for each of the existing sections.
    frame_urls: Optional[List[str]] = None

    #: Frames per second
    fps: Optional[float] = None

    #: Metadata of the slot
    metadata: Optional[Dict[str, UnknownType]] = None

    #: Frame Manifest for video slots
    frame_manifest: Optional[List[Dict[str, UnknownType]]] = None

    #: Segments for video slots
    segments: Optional[List[Dict[str, UnknownType]]] = None

    #: Upload ID
    upload_id: Optional[str] = None

    #: The reason for blocking upload of this slot, if it was blocked
    reason: Optional[str] = None


@dataclass
class SourceFile:
    #: File name of source file
    file_name: str

    #: URL of file
    url: Optional[str] = None


@dataclass
class AnnotationFileVersion:
    """
    Version of the AnnotationFile
    """

    major: int = 1
    minor: int = 0
    suffix: str = ""

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}{self.suffix}"


@dataclass
class Property:
    """
    Represents a property of an annotation file.
    """

    # Name of the property
    name: str

    # Type of the property
    type: PropertyType

    # Whether the property is required or not
    required: bool

    # Property options
    property_values: list[dict[str, Any]]

    # Granularity of the property
    granularity: PropertyGranularity

    # Description of the property
    description: Optional[str] = None

    # Granularity of the property
    granularity: PropertyGranularity = PropertyGranularity("section")


@dataclass
class PropertyClass:
    name: str
    type: str
    description: Optional[str]
    color: Optional[str] = None
    sub_types: Optional[list[str]] = None
    properties: Optional[list[Property]] = None


def parse_property_classes(metadata: dict[str, Any]) -> list[PropertyClass]:
    """
    Parses the metadata file and returns a list of PropertyClass objects.

    Parameters
    ----------
    metadata : dict[str, Any]
        The metadata file.

    Returns
    -------
    list[PropertyClass]
        A list of PropertyClass objects.
    """
    assert "classes" in metadata, "Metadata does not contain classes"

    classes = []
    for metadata_cls in metadata["classes"]:
        assert (
            "properties" in metadata_cls
        ), "Metadata class does not contain properties"
        properties = [
            Property(
                name=p["name"],
                type=p["type"],
                required=p["required"],
                property_values=p["property_values"],
                description=p.get("description"),
                granularity=PropertyGranularity(p.get("granularity", "section")),
            )
            for p in metadata_cls["properties"]
        ]
        classes.append(
            PropertyClass(
                name=metadata_cls["name"],
                type=metadata_cls["type"],
                description=metadata_cls.get("description"),
                color=metadata_cls.get("color"),
                sub_types=metadata_cls.get("sub_types"),
                properties=properties,
            )
        )
    return classes


def split_paths_by_metadata(
    path, dir: str = ".v7", filename: str = "metadata.json"
) -> tuple[Path, Optional[list[PropertyClass]]]:
    """
    Splits the given path into two: the path to the metadata file and the path to the properties

    Parameters
    ----------
    path : Path
        The path to the export directory.

    Returns
    -------
    tuple[Path, Optional[list[PropertyClass]]]
        A tuple containing the path to the metadata file and the list of property classes.
    """
    metadata_path = is_properties_enabled(path, dir, filename)
    if isinstance(metadata_path, bool):
        return path, None

    metadata = parse_metadata(metadata_path)
    property_classes = parse_property_classes(metadata)

    return metadata_path, property_classes


@dataclass
class AnnotationFile:
    """
    Represents a file containing annotations. Mostly useful when trying to import or export
    annotations to/from darwin V7.
    """

    #: Path to the file.
    path: Path

    #: Name of the file containing the annotations.
    filename: str

    #: ``Set`` of all ``AnnotationClass``\es this file contains.
    #: Used as a way to know in advance which ``AnnotationClass``\es this file has without having to
    #: go through the list of annotations.
    annotation_classes: Set[AnnotationClass]

    #: List of ``VideoAnnotation``\s or ``Annotation``\s.
    annotations: Sequence[Union[Annotation, VideoAnnotation]]

    # Item-level properties
    item_properties: Optional[list[dict[str, Any]]] = None

    # Deprecated
    #: Whether the annotations in the ``annotations`` attribute are ``VideoAnnotation`` or not.
    is_video: bool = False

    # Deprecated
    #: Width of the image in this annotation.
    image_width: Optional[int] = None

    # Deprecated
    #: Height of the image in this annotation.
    image_height: Optional[int] = None

    # Deprecated
    #: URL of the image in this annotation.
    image_url: Optional[str] = None

    #: URL of the workview for this annotation.
    workview_url: Optional[str] = None

    #: Sequence for this annotation.
    seq: Optional[int] = None

    # Deprecated
    #: URLs for the frames this ``AnnotationFile`` has.
    frame_urls: Optional[List[str]] = None

    #: Remote path for this ``Annotation``\'s file in V7's darwin.
    remote_path: Optional[str] = None

    slots: List[Slot] = field(default_factory=list)

    # Deprecated
    #: URL of the image's thumbnail in this annotation.
    image_thumbnail_url: Optional[str] = None

    #: Dataset name
    dataset_name: Optional[str] = None

    # Version of the file in format (MAJOR, MINOR, SUFFIX)
    # e.g. (1, 0, 'a')
    version: AnnotationFileVersion = field(default_factory=AnnotationFileVersion)

    # The darwin ID of the item that these annotations belong to.
    item_id: Optional[str] = None

    # The Frame Count if this is a video annotation
    frame_count: Optional[int] = None

    @property
    def full_path(self) -> str:
        """
        Returns the absolute path of this file.

        Returns
        -------
        str
            The absolute path of the file.
        """
        return construct_full_path(self.remote_path, self.filename)


def make_bounding_box(
    class_name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a bounding box annotation. ``x``, ``y``, ``w`` and ``h`` are rounded to 3
    decimal places when creating the annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.
    x : float
        The top left ``x`` value where the bounding box will start.
    y : float
        The top left ``y`` value where the bounding box will start.
    w : float
        The width of the bounding box.
    h : float
        The height of the bounding box.
    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``s for this ``Annotation``.

    Returns
    -------
    Annotation
        A bounding box ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "bounding_box"),
        {"x": round(x, 3), "y": round(y, 3), "w": round(w, 3), "h": round(h, 3)},
        subs or [],
        slot_names=slot_names or [],
    )


def make_tag(
    class_name: str,
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a tag annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.
    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``s for this ``Annotation``.

    Returns
    -------
    Annotation
        A tag ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "tag"), {}, subs or [], slot_names=slot_names or []
    )


def make_polygon(
    class_name: str,
    point_paths: List[List[Point]] | List[Point],
    bounding_box: Optional[Dict] = None,
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a polygon annotation.

    Parameters
    ----------
    class_name: str
        The name of the class for this ``Annotation``.
    point_paths: List[List[Point]] | List[Point]
        Either a list of points that comprises a polygon or a list of lists of points that comprises a complex polygon.
        A complex polygon is a polygon that is defined by >1 path.

        A polygon should be defined by a List[Point] and have a format similar to:

        ... code-block:: python

            [
                {"x": 1, "y": 0},
                {"x": 2, "y": 1}
            ]

        A complex polygon should be defined by a List[List[Point]] and have a format similar to:

        .. code-block:: python

            [
                [
                    {"x": 1, "y": 0},
                    {"x": 2, "y": 1}
                ],
                [
                    {"x": 3, "y": 4},
                    {"x": 5, "y": 6}
                ]
                # ... and so on ...
            ]

    bounding_box : Optional[Dict], default: None
        The bounding box that encompasses the polyong.
    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``s for this ``Annotation``.

    Returns
    -------
    Annotation
        A polygon ``Annotation``.
    """

    # Check if point_paths is List[Point] and convert to List[List[Point]]
    if (
        len(point_paths) > 1
        and isinstance(point_paths[0], dict)
        and "x" in point_paths[0]
        and "y" in point_paths[0]
    ):
        point_paths = [point_paths]

    return Annotation(
        AnnotationClass(class_name, "polygon", "polygon"),
        _maybe_add_bounding_box_data({"paths": point_paths}, bounding_box),
        subs or [],
        slot_names=slot_names or [],
    )


def make_complex_polygon(
    class_name: str,
    point_paths: List[List[Point]],
    bounding_box: Optional[Dict] = None,
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a complex polygon annotation. Complex polygons are those who have holes
    and/or disform shapes. This is used by the backend.

    Parameters
    ----------
    class_name: str
        The name of the class for this ``Annotation``.
    point_paths: List[List[Point]]
        A list of lists points that comprises the complex polygon. This is needed as a complex
        polygon can be effectively seen as a sum of multiple simple polygons. The list should have
        a format similar to:

        .. code-block:: python

            [
                [
                    {"x": 1, "y": 0},
                    {"x": 2, "y": 1}
                ],
                [
                    {"x": 3, "y": 4},
                    {"x": 5, "y": 6}
                ]
                # ... and so on ...
            ]

    bounding_box : Optional[Dict], default: None
        The bounding box that encompasses the polyong.
    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``s for this ``Annotation``.

    Returns
    -------
    Annotation
        A complex polygon ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "complex_polygon", "polygon"),
        _maybe_add_bounding_box_data({"paths": point_paths}, bounding_box),
        subs or [],
        slot_names=slot_names or [],
    )


def make_keypoint(
    class_name: str,
    x: float,
    y: float,
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a keypoint, aka point, annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.
    x : float
        The ``x`` value of the point.
    y : float
        The ``y`` value of the point.
    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``s for this ``Annotation``.

    Returns
    -------
    Annotation
        A point ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "keypoint"),
        {"x": x, "y": y},
        subs or [],
        slot_names=slot_names or [],
    )


def make_line(
    class_name: str,
    path: List[Point],
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a line annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.
    point_path : List[Point]
        A list of points that comprises the polygon. The list should have a format similar to:

        .. code-block:: python

            [
                {"x": 1, "y": 0},
                {"x": 2, "y": 1}
            ]

    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``s for this ``Annotation``.

    Returns
    -------
    Annotation
        A line ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "line"),
        {"path": path},
        subs or [],
        slot_names=slot_names or [],
    )


def make_skeleton(
    class_name: str,
    nodes: List[Node],
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a skeleton annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.
    nodes : List[Node]
        List of ``Node``\\s that comprise the skeleton. Each Node will have a format similar to:

        .. code-block:: python

            {
                "name": "1",
                "occluded": false,
                "x": 172.78,
                "y": 939.81
            }

    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``\\s for this ``Annotation``.

    Returns
    -------
    Annotation
        A skeleton ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "skeleton"),
        {"nodes": nodes},
        subs or [],
        slot_names=slot_names or [],
    )


def make_ellipse(
    class_name: str,
    parameters: EllipseData,
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns an Ellipse annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.
    parameters : EllipseData
        The data needed to build an Ellipse. This data must be a dictionary with a format similar
        to:

        .. code-block:: javascript

            {
                "angle": 0.57,
                "center": {
                    "x": 2745.69,
                    "y": 2307.46
                },
                "radius": {
                    "x": 467.02,
                    "y": 410.82
                }
            }

        Where:

        - ``angle: float`` is the orientation angle of the ellipse.
        - ``center: Point`` is the center point of the ellipse.
        - ``radius: Point`` is the width and height of the ellipse, where ``x`` represents the width and ``y`` represents height.

    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``s for this ``Annotation``.

    Returns
    -------
    Annotation
        An ellipse ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "ellipse"),
        parameters,
        subs or [],
        slot_names=slot_names or [],
    )


def make_cuboid(
    class_name: str,
    cuboid: CuboidData,
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a Cuboid annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.
    parameters : CuboidData
        The data needed to build a ``Cuboid``. This data must be a dictionary with a format similar
        to:

        .. code-block:: javascript

            {
                "back": {"h": 381.25, "w": 1101.81, "x": 1826.19, "y": 1841.44},
                "front": {"h": 575.69, "w": 1281.0, "x": 1742.31, "y": 1727.06}
            }

        Where:

        - ``back: Dict[str, float]`` is a dictionary containing the ``x`` and ``y`` of the top left corner Point, together with the width ``w`` and height ``h`` to form the back box.
        - ``front: Dict[str, float]`` is a dictionary containing the ``x`` and ``y`` of the top left corner Point, together with the width ``w`` and height ``h`` to form the front box.

    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``\\s for this ``Annotation``.

    Returns
    -------
    Annotation
        A cuboid ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "cuboid"),
        cuboid,
        subs or [],
        slot_names=slot_names or [],
    )


def make_table(
    class_name: str,
    bounding_box: BoundingBox,
    cells: List[Dict[str, UnknownType]],
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a table annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.

    bounding_box : BoundingBox
        Bounding box that wraps around the table.

    cells : List[Dict[str, Any]]
        Actual cells of the table. Their format should be similar to:

            .. code-block:: javascript

                [
                    {
                        "bounding_box": {
                            "h": 189.56,
                            "w": 416.37,
                            "x": 807.58,
                            "y": 1058.04
                        },
                        "col": 1,
                        "col_span": 1,
                        "id": "778691a6-0df6-4140-add9-f39806d950e9",
                        "is_header": false,
                        "row": 1,
                        "row_span": 1
                    }
                ]

    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``\\s for this ``Annotation``.

    Returns
    -------
    Annotation
        A table ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "table"),
        {"bounding_box": bounding_box, "cells": cells},
        subs or [],
        slot_names=slot_names or [],
    )


def make_simple_table(
    class_name: str,
    bounding_box: BoundingBox,
    col_offsets: List[float],
    row_offsets: List[float],
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a simple table annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.

    bounding_box : BoundingBox
        Bounding box that wraps around the table.

    col_offsets : List[float]
        List of floats representing the column offsets.

    row_offsets : List[float]
        List of floats representing the row offsets.

    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``\\s for this ``Annotation``.

    Returns
    -------
    Annotation
        A simple table ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "simple_table"),
        {
            "bounding_box": bounding_box,
            "col_offsets": col_offsets,
            "row_offsets": row_offsets,
        },
        subs or [],
        slot_names=slot_names or [],
    )


def make_string(
    class_name: str,
    sources: List[Dict[str, UnknownType]],
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a string annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.
    data : Any
        The data needed to build a ``String``. This data must be a list with a format similar
        to:

        .. code-block:: javascript

            [
                {
                    "id": "8cd598b5-0363-4984-9ae9-b15ccb77784a",
                    "ranges": [1, 2, 5]
                },
                {
                    "id": "6d6378d8-fd02-4518-8a21-6d94f0f32bbc",
                    "ranges": null
                }
            ]

    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``\\s for this ``Annotation``.

    Returns
    -------
    Annotation
        A string ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "string"),
        {"sources": sources},
        subs or [],
        slot_names=slot_names or [],
    )


def make_graph(
    class_name: str,
    nodes: List[Dict[str, str]],
    edges: List[Dict[str, str]],
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a graph annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.

    nodes : List[Dict[str, str]]
        Nodes of the graph. Should be in following format:
            .. code-block:: javascript

                [
                    {
                        "id": "91bb3c24-883a-433b-ae95-a6ee7845bea5",
                        "name": "key"
                    },
                    {
                        "id": "5a0ceba1-2e26-425e-8579-e6013ca415c5",
                        "name": "value"
                    }
                ]

    edges: List[Dict[str, str]]
        Edges of the graph. Should be in following format:
            .. code-block:: javascript

                [
                    {
                        "end": "value",
                        "start": "key"
                    }
                ]

    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``\\s for this ``Annotation``.

    Returns
    -------
    Annotation
        A graph ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "graph"),
        {"nodes": nodes, "edges": edges},
        subs or [],
        slot_names=slot_names or [],
    )


def make_mask(
    class_name: str,
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a mask annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.
    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``s for this ``Annotation``.

    Returns
    -------
    Annotation
        A mask ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "mask"), {}, subs or [], slot_names=slot_names or []
    )


def make_raster_layer(
    class_name: str,
    mask_annotation_ids_mapping: Dict[str, str],
    total_pixels: int,
    dense_rle: List[int],
    subs: Optional[List[SubAnnotation]] = None,
    slot_names: Optional[List[str]] = None,
) -> Annotation:
    """
    Creates and returns a raster_layer annotation.

    Parameters
    ----------
    class_name : str
        The name of the class for this ``Annotation``.

    mask_annotation_ids_mapping : Dict[str, str]
        Mapping of mask annotations ids to unique small integers used in the dense_rle.
        Should be in following format:
        .. code-block:: javascript

            {
                "91bb3c24-883a-433b-ae95-a6ee7845bea5": 1,
                "5a0ceba1-2e26-425e-8579-e6013ca415c5": 2
            }

    total_pixels : int
        Total number of pixels in a corresponding image.

    dense_rle : int
        Run length encoding of all masks in the raster layer.
        Should be in following format:
        .. code-block:: javascript

            [0, 5, 1, 15, 2, 10]

    subs : Optional[List[SubAnnotation]], default: None
        List of ``SubAnnotation``\\s for this ``Annotation``.

    Returns
    -------
    Annotation
        A raster_layer ``Annotation``.
    """
    return Annotation(
        AnnotationClass(class_name, "raster_layer"),
        {
            "mask_annotation_ids_mapping": mask_annotation_ids_mapping,
            "total_pixels": total_pixels,
            "dense_rle": dense_rle,
        },
        subs or [],
        slot_names=slot_names or [],
    )


def make_instance_id(value: int) -> SubAnnotation:
    """
    Creates and returns an instance id sub-annotation.

    Parameters
    ----------
    value : int
        The value of this instance's id.


    Returns
    -------
    SubAnnotation
        An instance id ``SubAnnotation``.
    """
    return SubAnnotation("instance_id", value)


def make_attributes(attributes: List[str]) -> SubAnnotation:
    """
    Creates and returns an attributes sub-annotation.

    Parameters
    ----------
    value : List[str]
        A list of attributes. Example: ``["orange", "big"]``.

    Returns
    -------
    SubAnnotation
        An attributes ``SubAnnotation``.
    """
    return SubAnnotation("attributes", attributes)


def make_text(text: str) -> SubAnnotation:
    """
    Creates and returns a text sub-annotation.

    Parameters
    ----------
    text : str
        The text for the sub-annotation.

    Returns
    -------
    SubAnnotation
        A text ``SubAnnotation``.
    """
    return SubAnnotation("text", text)


def make_opaque_sub(type: str, data: UnknownType) -> SubAnnotation:
    """
    Creates and returns a opaque sub-annotation.

    Parameters
    ----------
    type : str
        Type of this sub-annotation

    data : Any
        Data for this sub-annotation.

    Returns
    -------
    SubAnnotation
        A text ``SubAnnotation``.
    """
    return SubAnnotation(type, data)


KeyFrame = Dict[str, Union[int, Annotation]]


def make_keyframe(annotation: Annotation, idx: int) -> KeyFrame:
    """
    Creates and returns a ``KeyFrame``.

    Parameters
    ----------
    annotation : Annotation
        The annotation for the keyframe.
    idx : int
        The id of the keyframe.

    Returns
    -------
    KeyFrame
        The created ``Keyframe``.
    """
    return {"idx": idx, "annotation": annotation}


def make_video_annotation(
    frames: Dict[int, UnknownType],
    keyframes: Dict[int, bool],
    segments: List[Segment],
    interpolated: bool,
    slot_names: List[str],
    properties: Optional[list[SelectedProperty]] = None,
    hidden_areas: Optional[List[HiddenArea]] = None,
) -> VideoAnnotation:
    """
    Creates and returns a ``VideoAnnotation``.

    Parameters
    ----------
    frames : Dict[int, Any]
        The frames for the video. All frames must have the same ``annotation_class.name`` value.
    keyframes : Dict[int, bool]
        Indicates which frames are keyframes.
    segments : List[Segment]
        The list of segments for the video.
    interpolated : bool
        If this video annotation is interpolated or not.

    Returns
    -------
    VideoAnnotation
        The created ``VideoAnnotation``.

    Raises
    ------
    ValueError
        If some of the frames have different annotation class names.
    """
    first_annotation: Annotation = list(frames.values())[0]  # type: ignore
    if not all(frame.annotation_class.name == first_annotation.annotation_class.name for frame in frames.values()):  # type: ignore
        raise ValueError("invalid argument to make_video_annotation")

    return VideoAnnotation(
        first_annotation.annotation_class,
        frames,
        keyframes,
        segments,
        interpolated,
        slot_names=slot_names or [],
        properties=properties,
        hidden_areas=hidden_areas or [],
    )


def _maybe_add_bounding_box_data(
    data: Dict[str, UnknownType], bounding_box: Optional[Dict]
) -> Dict[str, UnknownType]:
    if bounding_box:
        data["bounding_box"] = {
            "x": bounding_box["x"],
            "y": bounding_box["y"],
            "w": bounding_box["w"],
            "h": bounding_box["h"],
        }
    return data


ExportParser = Callable[[Iterator[AnnotationFile], Path], None]

ImportParser = Callable[[Path], Union[List[AnnotationFile], AnnotationFile, None]]


class MaskTypes:
    Palette = Dict[str, int]
    Mode = Literal["index", "grey", "rgb"]
    TypeOfRender = Literal["raster", "polygon"]
    CategoryList = List[str]
    ExceptionList = List[Exception]
    UndecodedRLE = List[int]
    ColoursDict = Dict[str, int]
    RgbColors = List[int]
    HsvColors = List[Tuple[float, float, float]]
    RgbColorList = List[RgbColors]
    RgbPalette = Dict[str, RgbColors]

    RendererReturn = Tuple[ExceptionList, NDArray, CategoryList, ColoursDict]


@dataclass
class AnnotationMask:
    id: str
    name: str
    slot_names: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Mask name cannot be empty")
        if not self.slot_names:
            raise ValueError("Mask must be associated with at least one slot")
        if not self.id:
            raise ValueError("Mask ID cannot be empty")


@dataclass
class RasterLayer:
    rle: MaskTypes.UndecodedRLE
    mask_annotation_ids_mapping: Dict[str, int]
    slot_names: List[str] = field(default_factory=list)
    total_pixels: int = 0

    def validate(self) -> None:
        if not self.rle:
            raise ValueError("RasterLayer rle cannot be empty")
        if not self.mask_annotation_ids_mapping:
            raise ValueError("RasterLayer mask_annotation_ids_mapping cannot be empty")
        if not self.slot_names:
            raise ValueError("RasterLayer must be associated with at least one slot")
        if not self.total_pixels and not self.total_pixels > 0:
            raise ValueError("RasterLayer total_pixels cannot be empty")


@dataclass
class ManifestItem:
    frame: int
    absolute_frame: Optional[int]
    segment: int
    visibility: bool
    timestamp: float
    visible_frame: Optional[int]


@dataclass
class SegmentManifest:
    slot: str
    segment: int
    total_frames: int
    items: List[ManifestItem]


class ObjectStore:
    """
    Object representing a configured conection to an external storage locaiton

    Attributes:
        name (str): The alias of the storage connection
        prefix (str): The directory that files are written back to in the storage location
        readonly (bool): Whether the storage configuration is read-only or not
        provider (str): The cloud provider (aws, azure, or gcp)
        default (bool): Whether the storage connection is the default one
    """

    def __init__(
        self,
        name: str,
        prefix: str,
        readonly: bool,
        provider: str,
        default: bool,
    ) -> None:
        self.name = name
        self.prefix = prefix
        self.readonly = readonly
        self.provider = provider
        self.default = default

    def __str__(self) -> str:
        return f"Storage configuration:\n- Name: {self.name}\n- Prefix: {self.prefix}\n- Readonly: {self.readonly}\n- Provider: {self.provider}\n- Default: {self.default}"

    def __repr__(self) -> str:
        return f"ObjectStore(name={self.name}, prefix={self.prefix}, readonly={self.readonly}, provider={self.provider})"


class StorageKeyDictModel(BaseModel):
    storage_keys: Dict[str, List[str]]


class StorageKeyListModel(BaseModel):
    storage_keys: List[str]


class CartesianAxis(Enum):
    X = "x"
    Y = "y"
    Z = "z"
