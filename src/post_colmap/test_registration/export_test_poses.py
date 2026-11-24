
#!/usr/bin/env python3
# export_test_poses.py
"""
Exporta las poses de las imágenes de test registradas por COLMAP
al sistema físico FLU, reutilizando la transformación de v2_0.

No estima una transformación nueva.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]

# Permite importar las funciones del script original de alineación.
POST_COLMAP_DIR = ROOT / "src" / "post_colmap"
sys.path.insert(0, str(POST_COLMAP_DIR))

from align_colmap_to_flu import (  # noqa: E402
    R_BODY_FROM_CAMERA,
    pose_row,
    read_colmap_images,
)


MODEL_IMAGES = (
    ROOT / "results_colmap" / "v2_1_registration"
    / "sparse" / "0_txt" / "images.txt"
)

TRANSFORM_FILE = (
    ROOT / "results_colmap" / "v2_0" / "alignment"
    / "similarity_transform_colmap_to_FLU.txt"
)

TEST_LIST = (
    ROOT / "results_colmap" / "v2_1_registration"
    / "test_images.txt"
)

OUTPUT_DIR = (
    ROOT / "results_colmap" / "v2_1_registration" / "alignment"
)


def load_transform(path):
    """Lee la matriz homogénea y recupera escala, rotación y traslación."""

    matrix = np.loadtxt(path)

    if matrix.shape != (4, 4):
        raise ValueError(
            f"Se esperaba una matriz 4x4, se obtuvo {matrix.shape}"
        )

    A = matrix[:3, :3]
    translation = matrix[:3, 3]

    scale = float(np.cbrt(np.linalg.det(A)))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Escala inválida: {scale}")

    R_align = A / scale

    if not np.allclose(R_align.T @ R_align, np.eye(3), atol=1e-5):
        raise ValueError("La rotación recuperada no es ortogonal.")

    if not np.isclose(np.linalg.det(R_align), 1.0, atol=1e-5):
        raise ValueError(
            f"det(R_align) debe ser +1; valor: {np.linalg.det(R_align)}"
        )

    return scale, R_align, translation


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    for path in (MODEL_IMAGES, TRANSFORM_FILE, TEST_LIST):
        if not path.is_file():
            raise FileNotFoundError(f"No existe: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Leyendo poses del modelo COLMAP...")
    images = read_colmap_images(MODEL_IMAGES)

    print("Leyendo transformación original v2_0...")
    scale, R_align, translation = load_transform(TRANSFORM_FILE)

    print("Leyendo lista de imágenes de test...")
    test_names = [
        line.strip()
        for line in TEST_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(test_names) != len(set(test_names)):
        raise RuntimeError("Hay nombres duplicados en test_images.txt.")

    # El modelo puede contener rutas o nombres con espacios.
    # Para localizar las imágenes test usamos el nombre del archivo.
    images_by_basename = {}

    for name, record in images.items():
        basename = Path(name).name

        if basename in images_by_basename:
            raise RuntimeError(
                f"Nombre de imagen ambiguo en COLMAP: {basename}"
            )

        images_by_basename[basename] = (name, record)

    registered_rows = []
    unregistered_rows = []

    per_sequence = {}

    for test_name in test_names:
        basename = Path(test_name).name

        match = images_by_basename.get(basename)

        sequence_match = re.search(r"(bebop_test_\d+)", basename)
        sequence = (
            sequence_match.group(1)
            if sequence_match
            else "unknown"
        )

        stats = per_sequence.setdefault(
            sequence,
            {"total": 0, "registered": 0},
        )
        stats["total"] += 1

        if match is None:
            unregistered_rows.append({
                "image_name": basename,
                "sequence": sequence,
                "status": "not_registered",
            })
            continue

        _, record = match

        # Posición del centro de cámara en coordenadas COLMAP.
        center_colmap = record["center_colmap"]

        # Transformación métrica COLMAP -> FLU.
        position_flu = (
            scale * (R_align @ center_colmap) + translation
        )

        # COLMAP almacena world-to-camera.
        R_camera_world_colmap = (
            record["R_world_to_camera"].T
        )

        # Cámara -> mundo físico.
        R_camera_world_physical = (
            R_align @ R_camera_world_colmap
        )

        # Cámara óptica -> cuerpo FLU.
        R_world_body = (
            R_camera_world_physical @ R_BODY_FROM_CAMERA.T
        )

        row = pose_row(
            basename,
            record["image_id"],
            position_flu,
            R_world_body,
        )

        registered_rows.append(row)
        stats["registered"] += 1

    fields = [
        "image_name",
        "image_id",
        "x_m",
        "y_m",
        "z_m",
        "qx",
        "qy",
        "qz",
        "qw",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
    ]

    poses_path = OUTPUT_DIR / "poses_test_FLU.csv"
    unregistered_path = OUTPUT_DIR / "unregistered_test_images.csv"
    summary_path = OUTPUT_DIR / "test_registration_summary.txt"

    write_csv(poses_path, registered_rows, fields)

    write_csv(
        unregistered_path,
        unregistered_rows,
        ["image_name", "sequence", "status"],
    )

    total = len(test_names)
    registered = len(registered_rows)
    unregistered = len(unregistered_rows)
    coverage = 100.0 * registered / total if total else 0.0

    lines = [
        "REGISTRO Y EXPORTACIÓN DE POSES TEST",
        "====================================",
        "",
        f"Modelo COLMAP: {MODEL_IMAGES}",
        f"Transformación reutilizada: {TRANSFORM_FILE}",
        f"Imágenes en lista de test: {total}",
        f"Imágenes registradas: {registered}",
        f"Imágenes no registradas: {unregistered}",
        f"Cobertura: {coverage:.2f}%",
        "",
        f"Escala reutilizada (m/unidad COLMAP): {scale:.12f}",
        f"det(R_align): {np.linalg.det(R_align):.12f}",
        "",
        "COBERTURA POR SECUENCIA",
        "-----------------------",
    ]

    for sequence in sorted(per_sequence):
        stats = per_sequence[sequence]
        seq_coverage = (
            100.0 * stats["registered"] / stats["total"]
            if stats["total"] else 0.0
        )

        lines.append(
            f"{sequence}: "
            f"{stats['registered']}/{stats['total']} "
            f"({seq_coverage:.2f}%)"
        )

    lines.extend([
        "",
        "ARCHIVOS GENERADOS",
        "------------------",
        str(poses_path),
        str(unregistered_path),
        str(summary_path),
        "",
        "Convención:",
        "Posición: centro de cámara COLMAP transformado a FLU.",
        "Orientación: cuerpo FLU en el mundo físico.",
        "Cuaternión: qx, qy, qz, qw.",
        "Euler: RPY en grados, convención ZYX.",
        "No se estimó una transformación nueva.",
    ])

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\nEXPORTACIÓN COMPLETADA")
    print(f"Imágenes de test: {total}")
    print(f"Registradas: {registered}")
    print(f"No registradas: {unregistered}")
    print(f"Cobertura: {coverage:.2f}%")
    print(f"Escala: {scale:.9f} m/unidad COLMAP")
    print(f"Resultados: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
