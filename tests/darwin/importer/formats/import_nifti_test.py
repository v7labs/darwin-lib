import argparse
import json
import tempfile
from pathlib import Path
from typing import Union
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
from scipy import ndimage
import nibabel as nib

from darwin.datatypes import (
    Annotation,
    AnnotationClass,
    AnnotationFile,
    SubAnnotation,
    VideoAnnotation,
)
from darwin.importer.formats.nifti import get_new_axial_size, parse_path, process_nifti
from tests.fixtures import *
from darwin.utils.utils import parse_darwin_json


def test_image_annotation_nifti_import_single_slot(team_slug_darwin_json_v2: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmpdir)
            label_path = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti"
                / "releases"
                / "latest"
                / "annotations"
                / "vol0_brain.nii.gz"
            )
            input_dict = {
                "data": [
                    {
                        "image": "vol0 (1).nii",
                        "label": str(label_path),
                        "class_map": {"1": "brain"},
                        "mode": "video",
                    }
                ]
            }
            upload_json = Path(tmpdir) / "annotations.json"
            upload_json.write_text(
                json.dumps(input_dict, indent=4, sort_keys=True, default=str)
            )
            annotation_files = parse_path(path=upload_json)
            annotation_file = annotation_files[0]
            output_json_string = json.loads(
                serialise_annotation_file(annotation_file, as_dict=False)
            )
            expected_json_string = json.load(
                open(
                    Path(tmpdir)
                    / team_slug_darwin_json_v2
                    / "nifti"
                    / "vol0_annotation_file.json",
                    "r",
                )
            )
            assert (
                output_json_string["annotations"][0]["frames"]
                == expected_json_string["annotations"][0]["frames"]
            )


def test_image_annotation_nifti_import_mpr(team_slug_darwin_json_v2: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmpdir)
            label_path = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti"
                / "releases"
                / "latest"
                / "annotations"
                / "vol0_brain.nii.gz"
            )
            input_dict = {
                "data": [
                    {
                        "image": "vol0 (1).nii",
                        "label": str(label_path),
                        "class_map": {"1": "brain"},
                        "mode": "video",
                        "is_mpr": True,
                        "slot_names": ["0.3", "0.2", "0.1"],
                    }
                ]
            }
            upload_json = Path(tmpdir) / "annotations.json"
            upload_json.write_text(
                json.dumps(input_dict, indent=4, sort_keys=True, default=str)
            )
            remote_file_medical_metadata = {
                Path("/vol0 (1).nii"): {
                    "0.3": {
                        "affine": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        "original_affine": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        "pixdims": [1, 1, 1],
                        "primary_plane": "AXIAL",
                        "height": 100,
                        "width": 100,
                        "num_frames": 10,
                        "legacy": False,
                    },
                    "0.2": {
                        "affine": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        "original_affine": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        "pixdims": [1, 1, 1],
                        "primary_plane": "CORONAL",
                        "height": 100,
                        "width": 100,
                        "num_frames": 10,
                        "legacy": False,
                    },
                    "0.1": {
                        "affine": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        "original_affine": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        "pixdims": [1, 1, 1],
                        "primary_plane": "SAGITTAL",
                        "height": 100,
                        "width": 100,
                        "num_frames": 10,
                        "legacy": False,
                    },
                }
            }
            annotation_files = parse_path(
                path=upload_json,
                remote_file_medical_metadata=remote_file_medical_metadata,
            )
            annotation_file = annotation_files[0]
            output_json_string = json.loads(
                serialise_annotation_file(annotation_file, as_dict=False)
            )
            expected_json_string = json.load(
                open(
                    Path(tmpdir)
                    / team_slug_darwin_json_v2
                    / "nifti"
                    / "vol0_annotation_file_multi_slot.json",
                    "r",
                )
            )
            assert (
                output_json_string["annotations"][0]["frames"]
                == expected_json_string["annotations"][0]["frames"]
            )


def test_image_annotation_nifti_import_incorrect_number_slot(
    team_slug_darwin_json_v2: str,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmpdir)
            label_path = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti"
                / "releases"
                / "latest"
                / "annotations"
                / "vol0_brain.nii.gz"
            )
            input_dict = {
                "data": [
                    {
                        "image": "vol0 (1).nii",
                        "label": str(label_path),
                        "class_map": {"1": "brain"},
                        "mode": "video",
                        "is_mpr": True,
                        "slot_names": ["0.3", "0.2"],
                    }
                ]
            }
            upload_json = Path(tmpdir) / "annotations.json"
            upload_json.write_text(
                json.dumps(input_dict, indent=4, sort_keys=True, default=str)
            )
            with pytest.raises(Exception):
                parse_path(path=upload_json)


def test_image_annotation_nifti_import_single_slot_to_mask_legacy(
    team_slug_darwin_json_v2: str,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmpdir)
            label_path = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti"
                / "releases"
                / "latest"
                / "annotations"
                / "sample_nifti.nii"
            )
            input_dict = {
                "data": [
                    {
                        "image": "2044737.fat.nii",
                        "label": str(label_path),
                        "class_map": {"1": "brain"},
                        "mode": "mask",
                        "is_mpr": False,
                        "slot_names": ["0.1"],
                    }
                ]
            }
            upload_json = Path(tmpdir) / "annotations.json"
            upload_json.write_text(
                json.dumps(input_dict, indent=4, sort_keys=True, default=str)
            )

            with patch("darwin.importer.formats.nifti.zoom") as mock_zoom:
                mock_zoom.side_effect = ndimage.zoom
                remote_file_medical_metadata = {
                    Path("/2044737.fat.nii"): {
                        "0.1": {
                            "affine": [
                                [2.23214293, 0, 0, -247.76787233],
                                [0, 2.23214293, 0, -191.96429443],
                                [0, 0, 3, -21],
                                [0, 0, 0, 1],
                            ],
                            "original_affine": [
                                [2.23214293, 0, 0, -247.76787233],
                                [0, 2.23214293, 0, -191.96429443],
                                [0, 0, 3, -21],
                                [0, 0, 0, 1],
                            ],
                            "pixdims": [1, 1, 1],
                            "primary_plane": "AXIAL",
                            "height": 100,
                            "width": 100,
                            "num_frames": 10,
                            "legacy": True,
                        },
                    }
                }
                annotation_files = parse_path(
                    path=upload_json,
                    remote_file_medical_metadata=remote_file_medical_metadata,
                )
                annotation_file = annotation_files[0]
                output_json_string = json.loads(
                    serialise_annotation_file(annotation_file, as_dict=False)
                )
                expected_json_string = json.load(
                    open(
                        Path(tmpdir)
                        / team_slug_darwin_json_v2
                        / "nifti"
                        / "sample_nifti.nii.json",
                        "r",
                    )
                )
                # This needs to not check for mask_annotation_ids_mapping as these
                # are randomly generated
                [
                    frame.get("raster_layer", {}).pop("mask_annotation_ids_mapping")
                    for frame in output_json_string["annotations"][0]["frames"].values()
                ]
                [
                    frame.get("raster_layer", {}).pop("mask_annotation_ids_mapping")
                    for frame in expected_json_string["annotations"][1][
                        "frames"
                    ].values()
                ]

                assert (
                    mock_zoom.call_count
                    == expected_json_string["item"]["slots"][0]["frame_count"]
                )
                assert (
                    output_json_string["annotations"][0]["frames"]
                    == expected_json_string["annotations"][1]["frames"]
                )


def test_get_new_axial_size():
    volume = np.zeros((10, 10, 10))
    pixdims = (1, 0.5, 0.5)
    new_size = get_new_axial_size(volume, pixdims)
    assert new_size == (10, 10)


def test_get_new_axial_size_with_isotropic():
    volume = np.zeros((10, 10, 10))
    pixdims = (1, 0.5, 0.5)
    new_size = get_new_axial_size(volume, pixdims, isotropic=True)
    assert new_size == (20, 10)


def test_process_nifti_orientation_ras_to_lpi(team_slug_darwin_json_v2):
    """
    Test that an input NifTI annotation file in the RAS orientation is correctly
    transformed to the LPI orientation.

    Do this by emulating the `process_nifti` function, which:
    - 1: Transforms the input file into the RAS orientation
    - 2: Transforms the transformed RAS file into the LPI orientation
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmpdir)
            filepath = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti"
                / "releases"
                / "latest"
                / "annotations"
                / "vol0_brain.nii.gz"
            )
            lpi_ornt = [[0.0, -1.0], [1.0, -1.0], [2.0, -1.0]]
            ras_file = nib.load(filepath)
            ras_transformed_file = nib.funcs.as_closest_canonical(ras_file)
            lpi_transformed_file = nib.orientations.apply_orientation(
                ras_transformed_file.get_fdata(), lpi_ornt
            )
            processed_file = process_nifti(input_data=ras_file)
            assert not np.array_equal(processed_file, ras_file._dataobj)
            assert np.array_equal(processed_file, lpi_transformed_file)


def test_process_nifti_orientation_las_to_lpi(team_slug_darwin_json_v2):
    """
    Test that an input NifTI annotation file in the LAS orientation is correctly
    transformed to the LPI orientation.

    Do this by emulating the `process_nifti` function, which:
    - 1: Transforms the input file into the RAS orientation
    - 2: Transforms the transformed RAS file into the LPI orientation
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmpdir)
            filepath = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti"
                / "releases"
                / "latest"
                / "annotations"
                / "BRAINIX_NIFTI_ROI.nii.gz"
            )
            lpi_ornt = [[0.0, -1.0], [1.0, -1.0], [2.0, -1.0]]
            las_file = nib.load(filepath)
            ras_transformed_file = nib.funcs.as_closest_canonical(las_file)
            lpi_transformed_file = nib.orientations.apply_orientation(
                ras_transformed_file.get_fdata(), lpi_ornt
            )
            processed_file = process_nifti(input_data=las_file)
            assert not np.array_equal(processed_file, las_file._dataobj)
            assert np.array_equal(processed_file, lpi_transformed_file)


def serialise_annotation_file(
    annotation_file: AnnotationFile, as_dict
) -> Union[str, dict]:
    """
    Serialises an ``AnnotationFile`` into a string.

    Parameters
    ----------
    annotation_file : AnnotationFile
        The ``AnnotationFile`` to serialise.

    Returns
    -------
    str
        The serialised ``AnnotationFile``.
    """
    output_dict = {
        "path": str(annotation_file.path),
        "filename": annotation_file.filename,
        "annotation_classes": [
            serialise_annotation_class(ac, as_dict=True)
            for ac in annotation_file.annotation_classes
        ],
        "annotations": [
            serialise_annotation(a, as_dict=True) for a in annotation_file.annotations
        ],
        "is_video": annotation_file.is_video,
        "image_width": annotation_file.image_width,
        "image_height": annotation_file.image_height,
        "image_url": annotation_file.image_url,
        "workview_url": annotation_file.workview_url,
        "seq": annotation_file.seq,
        "frame_urls": annotation_file.frame_urls,
        "remote_path": annotation_file.remote_path,
    }

    json_string = json.dumps(
        output_dict,
        indent=4,
        sort_keys=True,
    )
    return output_dict if as_dict else json_string


def serialise_annotation(
    annotation: Union[Annotation, VideoAnnotation], as_dict
) -> Union[str, dict]:
    if isinstance(annotation, VideoAnnotation):
        return serialise_video_annotation(annotation, as_dict=as_dict)
    elif isinstance(annotation, Annotation):
        return serialise_general_annotation(annotation, as_dict=as_dict)


def serialise_general_annotation(annotation: Annotation, as_dict) -> Union[str, dict]:
    output_dict = {
        "annotation_class": annotation.annotation_class.name,
        "annotation_type": annotation.annotation_class.annotation_type,
        "data": annotation.data,
        "subs": [serialise_sub_annotation(sub) for sub in annotation.subs],
        "slot_names": annotation.slot_names,
    }
    json_string = json.dumps(
        output_dict,
        indent=4,
        sort_keys=True,
        default=str,
    )
    return output_dict if as_dict else json_string


def serialise_video_annotation(
    video_annotation: VideoAnnotation, as_dict: bool = True
) -> Union[str, dict]:
    data = video_annotation.get_data()
    output_dict = {
        "annotation_class": video_annotation.annotation_class.name,
        "annotation_type": video_annotation.annotation_class.annotation_type,
        "frames": data["frames"],
        "keyframes": video_annotation.keyframes,
        "segments": video_annotation.segments,
        "interpolated": video_annotation.interpolated,
        "slot_names": video_annotation.slot_names,
    }
    json_string = json.dumps(output_dict, indent=4, sort_keys=True, default=str)
    return output_dict if as_dict else json_string


def serialise_annotation_class(
    annotation_class: AnnotationClass, as_dict: bool = True
) -> Union[str, dict]:
    output_dict = {
        "name": annotation_class.name,
        "annotation_type": annotation_class.annotation_type,
        "annotation_internal_type": annotation_class.annotation_internal_type,
    }
    json_string = json.dumps(output_dict, indent=4, sort_keys=True, default=str)
    return output_dict if as_dict else json_string


def serialise_sub_annotation(
    sub_annotation: SubAnnotation, as_dict: bool = True
) -> Union[str, dict]:
    output_dict = {
        "type": sub_annotation.annotation_type,
        "data": sub_annotation.data,
    }
    json_string = json.dumps(
        output_dict,
        indent=4,
        sort_keys=True,
        default=str,
    )
    return output_dict if as_dict else json_string


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Update the serialisation of AnnotationFile with the current version."
    )
    input_json_string: str = """
    {
        "data": [
            {
                "image": "vol0 (1).nii",
                "label": "tests/v7/v7-darwin-json-v2/nifti/releases/latest/annotations/vol0_brain.nii.gz",
                "class_map": {
                    "1": "brain"
                },
                "is_mpr": true,
                "slot_names": ["0.3", "0.2", "0.1"],
                "mode": "video"
            }
        ]
    }
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "annotations.json"
        path.write_text(input_json_string)
        annotation_files = parse_path(path=path)
    if isinstance(annotation_files, list):
        annotation_file = annotation_files[0]
        output_json_string = serialise_annotation_file(annotation_file, as_dict=False)
        with open(
            Path("tests")
            / "v7"
            / "v7-darwin-json-v2"
            / "nifti"
            / "vol0_annotation_file_multi_slot.json",
            "w",
        ) as f:
            f.write(output_json_string)


def adjust_nifti_label_filepath(nifti_annotation_filepath: Path, nifti_filepath: Path):
    """
    Adjusts a specific NifTI label path to point to a local NifTI file for import testing.
    This is requied to allow the test to run in multiple environments
    """
    with open(nifti_annotation_filepath) as f:
        input_data = json.load(f)

    # Inject the nifti_filepath into the data.label key
    input_data["data"][0]["label"] = str(nifti_filepath)

    # Save the modified JSON back to the file
    with open(nifti_annotation_filepath, "w") as f:
        json.dump(input_data, f, indent=4)


def round_polygon_annotation_coordinates(annotation, decimal_places=2):
    """
    Rounds all coordinates in the annotation to a specified number of decimal places.

    Parameters:
    - annotation: The annotation data (list of lists).
    - decimal_places: The number of decimal places to round to.

    Returns:
    - A new annotation structure with rounded coordinates.
    """
    return [
        [
            {
                "x": round(point["x"], decimal_places),
                "y": round(point["y"], decimal_places),
            }
            for point in path
        ]
        for path in annotation
    ]


def test_parse_path_nifti_with_legacy_scaling():
    nifti_annotation_filepath = (
        Path(__file__).parents[2] / "data" / "nifti" / "nifti.json"
    )
    nifti_filepath = Path(__file__).parents[2] / "data" / "nifti" / "sample_nifti.nii"
    expected_annotations_filepath = (
        Path(__file__).parents[2]
        / "data"
        / "nifti"
        / "legacy"
        / "sample_nifti.nii.json"
    )
    adjust_nifti_label_filepath(nifti_annotation_filepath, nifti_filepath)
    expected_annotations = parse_darwin_json(expected_annotations_filepath)
    remote_file_medical_metadata = {
        Path("/2044737.fat.nii.gz"): {
            "0": {
                "affine": [
                    [2.23214293, 0, 0, -247.76787233],
                    [0, 2.23214293, 0, -191.96429443],
                    [0, 0, 3, -21],
                    [0, 0, 0, 1],
                ],
                "original_affine": [
                    [2.23214293, 0, 0, -247.76787233],
                    [0, 2.23214293, 0, -191.96429443],
                    [0, 0, 3, -21],
                    [0, 0, 0, 1],
                ],
                "pixdims": [1, 1, 1],
                "primary_plane": "AXIAL",
                "height": 100,
                "width": 100,
                "num_frames": 10,
                "legacy": True,
            },
        }
    }
    parsed_annotations = parse_path(
        nifti_annotation_filepath,
        remote_file_medical_metadata=remote_file_medical_metadata,
    )
    for frame_idx in expected_annotations.annotations[0].frames:
        expected_annotation = (
            expected_annotations.annotations[0].frames[frame_idx].data["paths"]
        )
        parsed_annotation = (
            parsed_annotations[0].annotations[0].frames[frame_idx].data["paths"]
        )
        expected_annotation_rounded = round_polygon_annotation_coordinates(
            expected_annotation, decimal_places=4
        )
        parsed_annotation_rounded = round_polygon_annotation_coordinates(
            parsed_annotation, decimal_places=4
        )
        assert expected_annotation_rounded == parsed_annotation_rounded


def test_parse_path_nifti_without_legacy_scaling():
    nifti_annotation_filepath = (
        Path(__file__).parents[2] / "data" / "nifti" / "nifti.json"
    )
    nifti_filepath = (
        Path(__file__).parents[2] / "data" / "nifti" / "BRAINIX_NIFTI_ROI.nii.gz"
    )
    expected_annotations_filepath = (
        Path(__file__).parents[2]
        / "data"
        / "nifti"
        / "no-legacy"
        / "BRAINIX_NIFTI_ROI.nii.json"
    )
    remote_file_medical_metadata = {
        Path("/2044737.fat.nii.gz"): {
            "0": {
                "affine": [
                    [2.23214293, 0, 0, -247.76787233],
                    [0, 2.23214293, 0, -191.96429443],
                    [0, 0, 3, -21],
                    [0, 0, 0, 1],
                ],
                "original_affine": [
                    [2.23214293, 0, 0, -247.76787233],
                    [0, 2.23214293, 0, -191.96429443],
                    [0, 0, 3, -21],
                    [0, 0, 0, 1],
                ],
                "pixdims": [0.7986109, 0.798611, 6.0000024],
                "primary_plane": "AXIAL",
                "height": 100,
                "width": 100,
                "num_frames": 10,
                "legacy": False,
            },
        }
    }
    adjust_nifti_label_filepath(nifti_annotation_filepath, nifti_filepath)
    expected_annotations = parse_darwin_json(expected_annotations_filepath)
    parsed_annotations = parse_path(
        nifti_annotation_filepath,
        remote_file_medical_metadata=remote_file_medical_metadata,
    )
    for frame_idx in expected_annotations.annotations[0].frames:
        expected_annotation = (
            expected_annotations.annotations[0].frames[frame_idx].data["paths"]
        )
        parsed_annotation = (
            parsed_annotations[0].annotations[0].frames[frame_idx].data["paths"]
        )
        expected_annotation_rounded = round_polygon_annotation_coordinates(
            expected_annotation, decimal_places=4
        )
        parsed_annotation_rounded = round_polygon_annotation_coordinates(
            parsed_annotation, decimal_places=4
        )
        assert expected_annotation_rounded == parsed_annotation_rounded
