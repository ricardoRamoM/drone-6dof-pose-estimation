#!/usr/bin/env python3
# filter_colmap_reconstruction.py	
# Filtrar reconstrucción de Colmap para eliminar poses inválidas.

"""
Filtra una reconstrucción COLMAP conservando exclusivamente
las imágenes del dataset actual de estimación de pose 6-DoF.

Entrada:
    results/colmap_benchmark_local_20261028/sparse_txt_1/
    results_colmap/v1_0_v2_0/sparse/1_txt/

Salida:
    results/colmap_filtered_1189/txt/
    results_colmap/v2_0/sparse/0_txt/

No modifica la reconstrucción original.

Uso
python src/post_colmap/filter/filter_colmap_reconstruction.py

"""

from pathlib import Path
from collections import Counter
import argparse
import re
import sys


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_MODEL = (
    PROJECT_ROOT
    / "results_colmap/v1_0_v2_0/sparse/1_txt/"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_colmap/v2_0/sparse/0_txt/"
)

EXPECTED_IMAGES = 1189
MIN_TRACK_LENGTH = 2

VIDEO_COUNTS = {
    "bebop_001_val": 305,
    "bebop_002_train": 138,
    "bebop_003_train": 217,
    "bebop_004_train": 238,
    "bebop_005_train": 276,
}

IMAGE_PATTERN = re.compile(
    r"^(ref_frame[0-9]+|"
    r"bebop_00[1-5]_(?:val|train)_frame_[0-9]+)\.jpg$"
)


# ============================================================
# LECTURA DE IMAGES.TXT
# ============================================================

def read_images(path):

    lines = path.read_text().splitlines()

    images = {}
    header = []

    i = 0

    while i < len(lines):

        line = lines[i]

        if not line.strip():
            i += 1
            continue

        if line.startswith("#"):
            header.append(line)
            i += 1
            continue

        fields = line.split(maxsplit=9)

        if len(fields) != 10:
            raise ValueError(
                f"Formato incorrecto en images.txt: {line}"
            )

        image_id = int(fields[0])
        name = fields[9]

        if image_id in images:
            raise ValueError(f"ID duplicado: {image_id}")

        if i + 1 >= len(lines):
            raise ValueError(
                f"Falta la línea de puntos 2D para {name}"
            )

        points_line = lines[i + 1]

        tokens = points_line.split()

        if len(tokens) % 3 != 0:
            raise ValueError(
                f"Formato incorrecto de puntos 2D: {name}"
            )

        points2D = []

        for j in range(0, len(tokens), 3):

            points2D.append([
                tokens[j],
                tokens[j + 1],
                int(tokens[j + 2])
            ])

        images[image_id] = {
            "line": line,
            "name": name,
            "points2D": points2D,
        }

        i += 2

    return header, images


# ============================================================
# LECTURA DE POINTS3D.TXT
# ============================================================

def read_points3D(path):

    lines = path.read_text().splitlines()

    points = {}
    header = []

    for line in lines:

        if not line.strip():
            continue

        if line.startswith("#"):
            header.append(line)
            continue

        fields = line.split()

        if len(fields) < 8:
            raise ValueError(
                f"Formato incorrecto en points3D.txt: {line}"
            )

        if (len(fields) - 8) % 2 != 0:
            raise ValueError(
                f"Track incorrecto: {line}"
            )

        point_id = int(fields[0])

        if point_id in points:
            raise ValueError(f"Punto 3D duplicado: {point_id}")

        track = []

        for i in range(8, len(fields), 2):

            track.append((
                int(fields[i]),
                int(fields[i + 1])
            ))

        points[point_id] = {
            "prefix": fields[:8],
            "track": track,
        }

    return header, points


# ============================================================
# VALIDACIÓN DEL MODELO ORIGINAL
# ============================================================

def validate_original(images, points):

    print("\nValidando reconstrucción original...")

    for point_id, point in points.items():

        for image_id, point2D_idx in point["track"]:

            if image_id not in images:
                raise ValueError(
                    f"Punto {point_id}: imagen inexistente {image_id}"
                )

            image = images[image_id]

            if point2D_idx >= len(image["points2D"]):
                raise ValueError(
                    f"Punto {point_id}: índice 2D inválido"
                )

            actual_id = image["points2D"][point2D_idx][2]

            if actual_id != point_id:
                raise ValueError(
                    f"Referencia inconsistente: "
                    f"imagen {image_id}, punto {point_id}"
                )

    print("Modelo original: referencias consistentes.")


# ============================================================
# IDENTIFICACIÓN DE IMÁGENES
# ============================================================

def select_images(images):

    selected = {}
    counts = Counter()

    for image_id, image in images.items():

        name = Path(image["name"]).name

        if IMAGE_PATTERN.fullmatch(name):

            selected[image_id] = image

            if name.startswith("ref_frame"):
                counts["references"] += 1

            else:
                video = "_".join(name.split("_")[:3])
                counts[video] += 1

    return selected, counts


# ============================================================
# FILTRADO DE PUNTOS 3D
# ============================================================

def filter_points(points, selected_image_ids):

    filtered = {}

    removed = 0

    for point_id, point in points.items():

        valid_track = [
            (image_id, point2D_idx)
            for image_id, point2D_idx in point["track"]
            if image_id in selected_image_ids
        ]

        if len(valid_track) >= MIN_TRACK_LENGTH:

            filtered[point_id] = {
                "prefix": point["prefix"],
                "track": valid_track,
            }

        else:
            removed += 1

    return filtered, removed


# ============================================================
# ACTUALIZACIÓN DE IMAGES.TXT
# ============================================================

def update_images(images, filtered_points):

    updated = {}

    valid_point_ids = set(filtered_points)

    for image_id, image in images.items():

        points2D = []

        for x, y, point_id in image["points2D"]:

            if point_id not in valid_point_ids:
                point_id = -1

            points2D.append([x, y, point_id])

        updated[image_id] = {
            "line": image["line"],
            "name": image["name"],
            "points2D": points2D,
        }

    return updated


# ============================================================
# VALIDACIÓN FINAL
# ============================================================

def validate_filtered(images, points):

    print("\nValidando reconstrucción filtrada...")

    selected_ids = set(images)

    for point_id, point in points.items():

        if len(point["track"]) < MIN_TRACK_LENGTH:
            raise ValueError(
                f"Punto {point_id} con track insuficiente"
            )

        for image_id, point2D_idx in point["track"]:

            if image_id not in selected_ids:
                raise ValueError(
                    f"Punto {point_id} referencia imagen excluida"
                )

            image = images[image_id]

            if point2D_idx >= len(image["points2D"]):
                raise ValueError(
                    f"Índice 2D inválido: {point_id}"
                )

            actual_id = image["points2D"][point2D_idx][2]

            if actual_id != point_id:
                raise ValueError(
                    f"Referencia cruzada incorrecta: {point_id}"
                )

    print("Referencias cruzadas: correctas.")


# ============================================================
# EXPORTACIÓN TXT
# ============================================================

def write_images(path, header, images):

    with path.open("w") as f:

        for line in header:
            f.write(line + "\n")

        for image_id in sorted(images):

            image = images[image_id]

            f.write(image["line"] + "\n")

            points_line = " ".join(
                f"{x} {y} {point_id}"
                for x, y, point_id in image["points2D"]
            )

            f.write(points_line + "\n")


def write_points3D(path, header, points):

    with path.open("w") as f:

        for line in header:
            f.write(line + "\n")

        for point_id in sorted(points):

            point = points[point_id]

            fields = point["prefix"].copy()

            for image_id, point2D_idx in point["track"]:
                fields.extend([
                    str(image_id),
                    str(point2D_idx)
                ])

            f.write(" ".join(fields) + "\n")


# ============================================================
# REPORTE
# ============================================================

def write_report(
    path,
    original_images,
    original_points,
    filtered_images,
    filtered_points,
    counts,
    removed_points
):

    track_lengths = Counter(
        len(point["track"])
        for point in filtered_points.values()
    )

    with path.open("w") as f:

        f.write("COLMAP FILTERING REPORT\n")
        f.write("======================\n\n")

        f.write(f"Imágenes originales: {len(original_images)}\n")
        f.write(f"Imágenes conservadas: {len(filtered_images)}\n")
        f.write(
            f"Imágenes excluidas: "
            f"{len(original_images) - len(filtered_images)}\n\n"
        )

        f.write(f"Puntos 3D originales: {len(original_points)}\n")
        f.write(f"Puntos 3D conservados: {len(filtered_points)}\n")
        f.write(f"Puntos 3D eliminados: {removed_points}\n\n")

        f.write("DISTRIBUCIÓN DE IMÁGENES\n")
        f.write("-----------------------\n")

        for name, count in sorted(counts.items()):
            f.write(f"{name}: {count}\n")

        f.write("\nDISTRIBUCIÓN DE TRACKS\n")
        f.write("---------------------\n")

        for length, count in sorted(track_lengths.items()):
            f.write(f"{length} observaciones: {count}\n")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_MODEL
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR
    )

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    print("=" * 60)
    print("COLMAP RECONSTRUCTION FILTER")
    print("=" * 60)

    for filename in [
        "cameras.txt",
        "images.txt",
        "points3D.txt"
    ]:

        if not (input_dir / filename).is_file():
            raise FileNotFoundError(
                f"No existe: {input_dir / filename}"
            )

    print("\nLeyendo modelo original...")

    cameras_path = input_dir / "cameras.txt"

    image_header, images = read_images(
        input_dir / "images.txt"
    )

    point_header, points = read_points3D(
        input_dir / "points3D.txt"
    )

    camera_text = cameras_path.read_text()

    print(f"Imágenes originales: {len(images)}")
    print(f"Puntos 3D originales: {len(points)}")

    validate_original(images, points)

    selected, counts = select_images(images)

    print("\nDistribución seleccionada:")

    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")

    if len(selected) != EXPECTED_IMAGES:
        raise ValueError(
            f"Se esperaban {EXPECTED_IMAGES} imágenes, "
            f"pero se encontraron {len(selected)}. "
            "No se exportará el modelo."
        )

    if counts["references"] != 15:
        raise ValueError(
            "El número de imágenes de referencia no es 15."
        )

    for video, expected in VIDEO_COUNTS.items():

        if counts[video] != expected:
            raise ValueError(
                f"{video}: se esperaban {expected}, "
                f"se encontraron {counts[video]}"
            )

    selected_ids = set(selected)

    print("\nFiltrando puntos 3D...")

    filtered_points, removed_points = filter_points(
        points,
        selected_ids
    )

    updated_images = update_images(
        selected,
        filtered_points
    )

    validate_filtered(
        updated_images,
        filtered_points
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"La carpeta de salida no está vacía: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "cameras.txt").write_text(camera_text)

    write_images(
        output_dir / "images.txt",
        image_header,
        updated_images
    )

    write_points3D(
        output_dir / "points3D.txt",
        point_header,
        filtered_points
    )

    report_dir = output_dir.parents[1]

    write_report(
        report_dir / "filtering_report.txt",
        images,
        points,
        updated_images,
        filtered_points,
        counts,
        removed_points
    )

    print("\n" + "=" * 60)
    print("FILTRADO COMPLETADO")
    print("=" * 60)

    print(f"Imágenes conservadas: {len(updated_images)}")
    print(f"Puntos 3D conservados: {len(filtered_points)}")
    print(f"Puntos 3D eliminados: {removed_points}")

    print(f"\nModelo TXT: {output_dir}")
    print(f"Reporte: {report_dir / 'filtering_report.txt'}")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)