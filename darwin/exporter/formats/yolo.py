from pathlib import Path
from typing import Dict, Iterable

import darwin.datatypes as dt

ClassIndex = Dict[str, int]


def export(annotation_files: Iterable[dt.AnnotationFile], output_dir: Path) -> None:
    """
    Exports the given ``AnnotationFile``\\s into the YOLO format inside of the given
    ``output_dir``.

    Parameters
    ----------
    annotation_files : Iterable[dt.AnnotationFile]
        The ``AnnotationFile``\\s to be exported.
    output_dir : Path
        The folder where the new pascalvoc files will be.
    """

    annotation_files = list(annotation_files)

    class_index = _build_class_index(annotation_files)

    for annotation_file in annotation_files:
        _export_file(annotation_file, class_index, output_dir)

    _save_class_index(class_index, output_dir)


def _export_file(annotation_file: dt.AnnotationFile, class_index: ClassIndex, output_dir: Path) -> None:
    txt = _build_txt(annotation_file, class_index)
    output_file_path = (output_dir / annotation_file.filename).with_suffix(".txt")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, "w") as f:
        f.write(txt)


def _build_class_index(annotation_files: Iterable[dt.AnnotationFile]) -> ClassIndex:
    index: ClassIndex = {}
    for annotation_file in annotation_files:
        for annotation in annotation_file.annotations:
            class_name = annotation.annotation_class.name
            if class_name not in index:
                index[class_name] = len(index)
    return index


def _build_txt(annotation_file: dt.AnnotationFile, class_index: ClassIndex) -> str:
    yolo_lines = []
    for annotation in annotation_file.annotations:
        annotation_type = annotation.annotation_class.annotation_type
        if annotation_type == "bounding_box":
            data = annotation.data
        elif annotation_type in ["polygon", "complex_polygon"]:
            data = annotation.data
            data = data.get("bounding_box")
        else:
            continue

        i = class_index[annotation.annotation_class.name]
        x = round(data.get("x"))
        y = round(data.get("y"))
        w = round(data.get("w"))
        h = round(data.get("h"))

        yolo_lines.append(f"{i} {x} {y} {w} {h}")
    return "\n".join(yolo_lines)


def _save_class_index(class_index: ClassIndex, output_dir: Path) -> None:
    with open(output_dir / "darknet.labels", "w") as f:
        for class_name in class_index:
            f.write(f"{class_name}\n")
