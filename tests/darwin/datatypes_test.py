import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, cast

import pytest

from darwin.datatypes import (
    make_raster_layer,
)
from darwin.client import Client
from darwin.config import Config
from darwin.dataset.remote_dataset_v2 import RemoteDatasetV2
from darwin.importer.importer import (
    _apply_axial_flips_to_annotations,
    _flip_annotation_in_z,
)
from darwin.datatypes import (
    ObjectStore,
    Point,
    make_polygon,
    make_bounding_box,
    make_video_annotation,
    parse_property_classes,
    split_paths_by_metadata,
    VideoAnnotation,
    AnnotationFile,
    CartesianAxis,
)


class TestAnnotationFlips:
    @pytest.fixture
    def bounding_box_annotation(self):
        return make_bounding_box(
            "test_box", x=10, y=20, w=30, h=40, slot_names=["slot1"]
        )

    @pytest.fixture
    def video_annotation(self, bounding_box_annotation):
        frames = {0: bounding_box_annotation, 10: bounding_box_annotation}
        keyframes = {0: True, 10: True}
        segments = [[0, 10]]
        return make_video_annotation(
            frames=frames,
            keyframes=keyframes,
            segments=segments,
            interpolated=True,
            slot_names=["slot1"],
        )

    @pytest.fixture
    def raster_layer_annotation(self):
        return make_raster_layer(
            "test_raster_layer",
            mask_annotation_ids_mapping={},
            total_pixels=8,
            dense_rle=[0, 2, 1, 2, 0, 2, 1, 2],
            slot_names=["slot1"],
        )

    def test_flip_bounding_box_x(self, bounding_box_annotation):
        """Test flipping bounding box in X axis"""
        bounding_box_annotation._flip_annotation_in_x_or_y(100, CartesianAxis.X)
        assert bounding_box_annotation.data["x"] == 90

    def test_flip_bounding_box_y(self, bounding_box_annotation):
        """Test flipping bounding box in Y axis"""
        bounding_box_annotation._flip_annotation_in_x_or_y(100, CartesianAxis.Y)
        assert bounding_box_annotation.data["y"] == 80

    def test_flip_polygon_x(self):
        """Test flipping polygon in X axis"""
        points: List[Dict[str, float]] = [
            {"x": 10.0, "y": 20.0},
            {"x": 30.0, "y": 40.0},
            {"x": 50.0, "y": 60.0},
        ]
        polygon = make_polygon("test_polygon", [points], slot_names=["slot1"])
        polygon._flip_annotation_in_x_or_y(100, CartesianAxis.X)
        assert polygon.data["paths"][0][0]["x"] == 90
        assert polygon.data["paths"][0][1]["x"] == 70
        assert polygon.data["paths"][0][2]["x"] == 50

    def test_flip_raster_layer_x(self, raster_layer_annotation):
        """Test flipping raster layer in X axis"""
        raster_layer_annotation._flip_raster_layer_in_x_or_y(4, 2, CartesianAxis.X)
        expected_pattern = [
            1,
            2,
            0,
            2,
            1,
            2,
            0,
            2,
        ]
        assert raster_layer_annotation.data["dense_rle"] == expected_pattern

    def test_flip_raster_layer_y(self, raster_layer_annotation):
        """Test flipping raster layer in Y axis"""
        raster_layer_annotation._flip_raster_layer_in_x_or_y(4, 2, CartesianAxis.Y)
        expected_pattern = [
            0,
            2,
            1,
            2,
            0,
            2,
            1,
            2,
        ]
        assert raster_layer_annotation.data["dense_rle"] == expected_pattern

    def test_flip_video_annotation_in_z(self, video_annotation):
        """Test flipping video annotation in Z axis"""

        flipped = cast(
            VideoAnnotation, _flip_annotation_in_z(video_annotation, num_frames=20)
        )
        assert 19 in flipped.frames
        assert 9 in flipped.frames
        assert flipped.segments == [[10, 20]]

    def test_flip_single_annotation_in_z(self, bounding_box_annotation):
        """Test that single annotations are returned unchanged when flipping in Z"""
        flipped = _flip_annotation_in_z(bounding_box_annotation, num_frames=20)
        assert flipped == bounding_box_annotation

    def test_apply_axial_flips(self, video_annotation):
        """Test applying all axial flips to video annotations"""
        medical_metadata = {
            Path("/test/path.json"): {
                "slot1": {
                    "legacy": False,
                    "axial_flips": [-1, -1, -1],
                    "width": 100,
                    "height": 200,
                    "num_frames": 20,
                }
            }
        }

        parsed_files = [
            AnnotationFile(
                path=Path("/test"),
                filename="path.json",
                annotation_classes=set(),
                annotations=[video_annotation],
                remote_path="/test",
            )
        ]

        flipped_files = _apply_axial_flips_to_annotations(
            parsed_files, medical_metadata
        )
        flipped_annotation = cast(VideoAnnotation, flipped_files[0].annotations[0])

        assert flipped_annotation.frames[19].data["x"] == 90
        assert flipped_annotation.frames[19].data["y"] == 180
        assert 19 in flipped_annotation.frames
        assert 9 in flipped_annotation.frames

    def test_apply_axial_flips_legacy_data(self, video_annotation):
        """Test that legacy data is not flipped"""

        medical_metadata = {
            Path("/test/path.json"): {
                "slot1": {
                    "legacy": True,
                    "axial_flips": [-1, -1, -1],
                    "width": 100,
                    "height": 200,
                    "num_frames": 20,
                }
            }
        }

        parsed_files = [
            AnnotationFile(
                path=Path("/test"),
                filename="path.json",
                annotation_classes=set(),
                annotations=[video_annotation],
                remote_path="/test/path.json",
            )
        ]

        flipped_files = _apply_axial_flips_to_annotations(
            parsed_files, medical_metadata
        )
        unflipped_annotation = cast(VideoAnnotation, flipped_files[0].annotations[0])

        assert unflipped_annotation.frames[0].data["x"] == 10
        assert unflipped_annotation.frames[0].data["y"] == 20
        assert 0 in unflipped_annotation.frames
        assert 10 in unflipped_annotation.frames


class TestMakePolygon:
    def test_it_returns_annotation_with_default_params(self):
        class_name: str = "class_name"
        points: List[Point] = [[{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}]]
        annotation = make_polygon(class_name, points)

        assert_annotation_class(annotation, class_name, "polygon", "polygon")

        paths = annotation.data.get("paths")
        assert paths == points

    def test_it_returns_annotation_with_bounding_box(self):
        class_name: str = "class_name"
        points: List[Point] = [[{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}]]
        bbox: Dict[str, float] = {"x": 1, "y": 2, "w": 2, "h": 2}
        annotation = make_polygon(class_name, points, bbox)

        assert_annotation_class(annotation, class_name, "polygon", "polygon")

        paths = annotation.data.get("paths")
        assert paths == points

        class_bbox = annotation.data.get("bounding_box")
        assert class_bbox == bbox


class TestMakeComplexPolygon:
    def test_it_returns_annotation_with_default_params(self):
        class_name: str = "class_name"
        points: List[List[Point]] = [
            [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}],
            [{"x": 4, "y": 5}, {"x": 6, "y": 7}, {"x": 4, "y": 5}],
        ]
        annotation = make_polygon(class_name, points)

        assert_annotation_class(annotation, class_name, "polygon", "polygon")

        paths = annotation.data.get("paths")
        assert paths == points

    def test_it_returns_annotation_with_bounding_box(self):
        class_name: str = "class_name"
        points: List[List[Point]] = [
            [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}],
            [{"x": 4, "y": 5}, {"x": 6, "y": 7}, {"x": 4, "y": 5}],
        ]
        bbox: Dict[str, float] = {"x": 1, "y": 2, "w": 2, "h": 2}
        annotation = make_polygon(class_name, points, bbox)

        assert_annotation_class(annotation, class_name, "polygon", "polygon")

        paths = annotation.data.get("paths")
        assert paths == points

        class_bbox = annotation.data.get("bounding_box")
        assert class_bbox == bbox


def assert_annotation_class(annotation, name, type, internal_type=None) -> None:
    assert annotation.annotation_class.name == name
    assert annotation.annotation_class.annotation_type == type
    assert annotation.annotation_class.annotation_internal_type == internal_type


@pytest.mark.parametrize(
    ("filename", "property_class_n", "properties_n"),
    (
        ("metadata.json", 1, [2]),
        ("metadata_nested_properties.json", 3, [0, 0, 1]),
        ("metadata_empty_properties.json", 3, [0, 0, 0]),
    ),
)
def test_parse_properties(filename, property_class_n, properties_n):
    metadata_path = Path(__file__).parent / f"data/{filename}"

    with open(metadata_path) as f:
        metadata = json.load(f)

    property_classes = parse_property_classes(metadata)
    assert len(property_classes) == property_class_n
    assert [
        len(property_class.properties or []) for property_class in property_classes
    ] == properties_n


@pytest.mark.parametrize(
    ("filename", "property_class_n", "properties_n", "is_properties_enabled"),
    (
        ("metadata.json", 1, [2], True),
        ("metadata_nested_properties.json", 3, [0, 0, 1], True),
        ("metadata_empty_properties.json", 0, [], False),
    ),
)
def test_split_paths_by_manifest(
    filename, property_class_n, properties_n, is_properties_enabled
):
    manifest_path = Path(__file__).parent / f"data/{filename}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmpdir_v7 = tmpdir / ".v7"
        tmpdir_v7.mkdir(exist_ok=True)
        shutil.copy(manifest_path, tmpdir_v7)

        _path, property_classes = split_paths_by_metadata(tmpdir, filename=filename)

        is_path_file = _path.is_file()
        assert is_path_file == is_properties_enabled
        assert len(property_classes or []) == property_class_n
        assert [
            len(property_class.properties or [])
            for property_class in property_classes or []
        ] == properties_n


class TestObjectStore:
    @pytest.fixture
    def object_store(self):
        return ObjectStore(
            name="test",
            prefix="test_prefix",
            readonly=False,
            provider="aws",
            default=True,
        )

    @pytest.fixture
    def darwin_client(
        darwin_config_path: Path,
        darwin_datasets_path: Path,
        team_slug_darwin_json_v2: str,
    ) -> Client:
        config = Config(darwin_config_path)
        config.put(["global", "api_endpoint"], "http://localhost/api")
        config.put(["global", "base_url"], "http://localhost")
        config.put(["teams", team_slug_darwin_json_v2, "api_key"], "mock_api_key")
        config.put(
            ["teams", team_slug_darwin_json_v2, "datasets_dir"],
            str(darwin_datasets_path),
        )
        return Client(config)

    @pytest.fixture
    def remote_dataset_v2(self):
        return RemoteDatasetV2(
            client=self.darwin_client,
            team="test_team",
            name="Test dataset",
            slug="test-dataset",
            dataset_id=1,
        )

    def test_init(self, object_store):
        assert object_store.name == "test"
        assert object_store.prefix == "test_prefix"
        assert object_store.readonly is False
        assert object_store.provider == "aws"
        assert object_store.default is True

    def test_str(self, object_store):
        assert (
            str(object_store)
            == "Storage configuration:\n- Name: test\n- Prefix: test_prefix\n- Readonly: False\n- Provider: aws\n- Default: True"
        )

    def test_repr(self, object_store):
        assert (
            repr(object_store)
            == "ObjectStore(name=test, prefix=test_prefix, readonly=False, provider=aws)"
        )
