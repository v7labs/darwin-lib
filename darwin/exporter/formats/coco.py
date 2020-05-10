import json
from datetime import date
from pathlib import Path
from typing import Generator, List

import numpy as np

import darwin.datatypes as dt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def export(
    annotation_files: Generator[dt.AnnotationFile, None, None], output_dir: Path
):
    output = build_json(list(annotation_files))
    # TODO, maybe an optional output name (like the dataset name if available)
    output_file_path = (output_dir / "output").with_suffix(".json")
    with open(output_file_path, "w") as f:
        json.dump(output, f, cls=NumpyEncoder, indent=1)


def build_json(annotation_files):
    categories = calculate_categories(annotation_files)
    return {
        "info": build_info(),
        "licenses": build_licenses(),
        "images": build_images(annotation_files),
        "annotations": list(build_annotations(annotation_files, categories)),
        "categories": list(build_categories(categories)),
    }


def calculate_categories(annotation_files: List[dt.AnnotationFile]):
    categories = {}
    for annotation_file in annotation_files:
        for annotation_class in annotation_file.annotation_classes:
            if (
                annotation_class.name not in categories
                and annotation_class.annotation_type == "polygon"
            ):
                categories[annotation_class.name] = len(categories)
    return categories


def build_info():
    # TODO fill out these fields in a meaningful way
    today = date.today()
    return {
        "description": "Exported from Darwin",
        "url": "n/a",
        "version": "n/a",
        "year": today.year,
        "contributor": "n/a",
        "date_created": today.strftime("%Y/%m/%d"),
    }


def build_licenses():
    return []


def build_images(annotation_files):
    return [
        build_image(id, annotation_file)
        for id, annotation_file in enumerate(annotation_files)
    ]


def build_image(id, annotation_file):
    return {
        "license": 0,
        "file_name": annotation_file.filename,
        "coco_url": "n/a",
        "height": annotation_file.image_height,
        "width": annotation_file.image_width,
        "date_captured": "",
        "flickr_url": "n/a",
        "darwin_url": annotation_file.image_url,
        "darwin_workview_url": annotation_file.workview_url,
        "id": id,
    }


def build_annotations(annotation_files, categories):
    annotation_id = 0
    for (image_id, annotation_file) in enumerate(annotation_files):
        for annotation in annotation_file.annotations:
            annotation_id += 1
            annotation_data = build_annotation(
                image_id, annotation_id, annotation, categories
            )
            if annotation_data:
                yield annotation_data


def build_annotation(image_id, annotation_id, annotation: dt.Annotation, categories):
    annotation_type = annotation.annotation_class.annotation_type
    if annotation_type == "polygon":
        sequences = convert_polygons_to_sequences(annotation.data["path"])
        x_coords = [s[0::2] for s in sequences]
        y_coords = [s[1::2] for s in sequences]
        min_x = np.min([np.min(x_coord) for x_coord in x_coords])
        min_y = np.min([np.min(y_coord) for y_coord in y_coords])
        max_x = np.max([np.max(x_coord) for x_coord in x_coords])
        max_y = np.max([np.max(y_coord) for y_coord in y_coords])
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        # Compute the area of the polygon
        poly_area = np.sum(
            [
                polygon_area(x_coord, y_coord)
                for x_coord, y_coord in zip(x_coords, y_coords)
            ]
        )

        data = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": categories[annotation.annotation_class.name],
            "segmentation": sequences,
            "area": poly_area,
            "bbox": [min_x, min_y, w, h],
            "iscrowd": 0,
        }
        instance_id_sub = annotation.get_sub("instance_id")
        if instance_id_sub:
            data["instance_id"] = instance_id_sub.data["value"]
        return data
    else:
        print(f"skipping unsupported annotation_type '{annotation_type}'")


def build_categories(categories):
    for id, name in categories.items():
        yield {"id": id, "name": name, "supercategory": "root"}


def convert_polygons_to_sequences(polygons: List) -> List[np.ndarray]:
    """
    Converts a list of polygons, encoded as a list of dictionaries of into a list of nd.arrays
    of coordinates.

    Parameters
    ----------
    polygons: list
        List of coordinates in the format [{x: x1, y:y1}, ..., {x: xn, y:yn}] or a list of them
        as  [[{x: x1, y:y1}, ..., {x: xn, y:yn}], ..., [{x: x1, y:y1}, ..., {x: xn, y:yn}]].

    Returns
    -------
    sequences: list[ndarray[float]]
        List of arrays of coordinates in the format [[x1, y1, x2, y2, ..., xn, yn], ...,
        [x1, y1, x2, y2, ..., xn, yn]]
    """
    if not polygons:
        raise ValueError("No polygons provided")
    # If there is a single polygon composing the instance the format is going to be
    # polygons = [{x: x1, y:y1}, ..., {x: xn, y:yn}]
    if isinstance(polygons[0], dict):
        path = []
        for point in polygons:
            path.append(point["x"])
            path.append(point["y"])
        return [np.array(path)]  # List type is used for backward compatibility
    # If there are multiple polygons composing the instance the format is going to be
    # polygons =  [[{x: x1, y:y1}, ..., {x: xn, y:yn}], ..., [{x: x1, y:y1}, ..., {x: xn, y:yn}]]
    if isinstance(polygons[0], list) and isinstance(polygons[0][0], dict):
        sequences = []
        for polygon in polygons:
            path = []
            for point in polygon:
                path.append(point["x"])
                path.append(point["y"])
            sequences.append(np.array(path))
        return sequences
    raise ValueError("Unknown input format")


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """
    Returns the area of the input polygon, represented with two numpy arrays
    for x and y coordinates.
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
