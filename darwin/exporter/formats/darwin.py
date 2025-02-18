from typing import Any, Dict, List


import darwin.datatypes as dt

# from darwin.datatypes import PolygonPath, PolygonPaths

DEPRECATION_MESSAGE = """

This function is going to be turned into private. This means that breaking 
changes in its interface and implementation are to be expected. We encourage using ``build_image_annotation`` 
instead of calling this low-level function directly.

"""


def build_image_annotation(
    annotation_file: dt.AnnotationFile, team_name: str
) -> Dict[str, Any]:
    """
    Builds and returns a dictionary with the annotations present in the given file in Darwin v2 format.

    Parameters
    ----------
    annotation_file: AnnotationFile
        File with the image annotations to extract.
        For schema, see: https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json

    Returns
    -------
    Dict[str, Any]
        A dictionary with the annotations in Darwin v2 format.
    """
    annotations_list: List[Dict[str, Any]] = []

    for annotation in annotation_file.annotations:
        annotation_data = _build_v2_annotation_data(annotation)
        annotations_list.append(annotation_data)

    slots_data = _build_slots_data(annotation_file.slots)
    item = _build_item_data(annotation_file, team_name)
    item["slots"] = slots_data

    return {
        "version": "2.0",
        "schema_ref": "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
        "item": item,
        "annotations": annotations_list,
    }


def _build_v2_annotation_data(annotation: dt.Annotation) -> Dict[str, Any]:
    annotation_data = {"id": annotation.id, "name": annotation.annotation_class.name}

    if annotation.annotation_class.annotation_type == "bounding_box":
        annotation_data["bounding_box"] = _build_bounding_box_data(annotation.data)
    elif annotation.annotation_class.annotation_type == "tag":
        annotation_data["tag"] = {}
    elif annotation.annotation_class.annotation_type == "polygon":
        polygon_data = _build_polygon_data(annotation.data)
        annotation_data["polygon"] = polygon_data
        annotation_data["bounding_box"] = _build_bounding_box_data(annotation.data)
    elif annotation.annotation_class.annotation_type == "keypoint":
        annotation_data["keypoint"] = annotation.data

    return annotation_data


def _build_bounding_box_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if "bounding_box" in data:
        data = data["bounding_box"]
    return {
        "h": data.get("h"),
        "w": data.get("w"),
        "x": data.get("x"),
        "y": data.get("y"),
    }


def _build_polygon_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the polygon data for Darwin V2 format from Darwin internal format (looks like V1).

    Parameters
    ----------
    data : Dict[str, Any]
        The original data for the polygon annotation.

    Returns
    -------
    Dict[str, List[List[Dict[str, float]]]]
        The polygon data in the format required for Darwin v2 annotations.
    """
    return {"paths": data["paths"]}


def _build_item_data(
    annotation_file: dt.AnnotationFile, team_name: str
) -> Dict[str, Any]:
    """
    Constructs the 'item' section of the Darwin v2 format annotation.

    Parameters
    ----------
    annotation_file: dt.AnnotationFile
        The AnnotationFile object containing annotation data.

    Returns
    -------
    Dict[str, Any]
        The 'item' section of the Darwin v2 format annotation.
    """
    return {
        "name": annotation_file.filename,
        "path": annotation_file.remote_path or "/",
        "source_info": {
            "dataset": {
                "name": annotation_file.dataset_name,
                "slug": (
                    annotation_file.dataset_name.lower().replace(" ", "-")
                    if annotation_file.dataset_name
                    else None
                ),
            },
            "item_id": annotation_file.item_id,
            "team": {
                "name": team_name,
                "slug": team_name.lower().replace(" ", "-"),
            },
            "workview_url": annotation_file.workview_url,
        },
    }


def _build_slots_data(slots: List[dt.Slot]) -> List[Dict[str, Any]]:
    """
    Constructs the 'slots' data for the Darwin v2 format annotation.

    Parameters
    ----------
    slots: List[Slot]
        A list of Slot objects from the AnnotationFile.

    Returns
    -------
    List[Dict[str, Any]]
        The 'slots' data for the Darwin v2 format annotation.
    """
    slots_data = []
    for slot in slots:
        slot_data = {
            "type": slot.type,
            "slot_name": slot.name,
            "width": slot.width,
            "height": slot.height,
            "thumbnail_url": slot.thumbnail_url,
            "source_files": slot.source_files,
        }
        slots_data.append(slot_data)

    return slots_data
