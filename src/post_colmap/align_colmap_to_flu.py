#!/usr/bin/env python3
# align_colmap_to_flu.py
"""Align COLMAP camera poses to the physical FLU frame and validate references.

Default inputs:
  results_colmap/v2_0/txt/images.txt
  dataset/v2_0/colmap_reference_positions.txt

Default outputs (written to results_colmap/v2_0/alignment/):
  poses_drone_FLU.csv
  alignment_summary.txt
  similarity_transform_colmap_to_FLU.txt

Coordinate conventions:
  Physical world: FLU (X forward, Y left, Z up).
  COLMAP camera: X right, Y down, Z forward.
  COLMAP image quaternions are world-to-camera (qw, qx, qy, qz).
  Camera position is used as the drone position (zero camera/body translation).

The script estimates a proper 3D similarity transform (scale, rotation,
translation) from all reference correspondences using Umeyama/Kabsch.
It does not modify the original COLMAP reconstruction.

python src/post_colmap/align_colmap_to_flu.py
python src/post_colmap/align_colmap_to_flu.py --help
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


DEFAULT_IMAGES = Path("results_colmap/v2_0/sparse/0_txt/images.txt")
DEFAULT_REFERENCES = Path("dataset/v2_0/colmap_reference_positions.txt")
DEFAULT_OUTPUT = Path("results_colmap/v2_0/alignment")

# Maps camera-coordinate vector components [right, down, forward]
# to body FLU components [forward, left, up].
R_BODY_FROM_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Se encontró un cuaternión de norma cero.")
    qw, qx, qy, qz = q / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def rotation_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Return quaternion in (qw, qx, qy, qz) order."""
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1e-15, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1e-15, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(1e-15, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    # q and -q represent the same rotation. Keep qw non-negative for consistency.
    if q[0] < 0:
        q *= -1
    return tuple(float(v) for v in q)


def rotation_to_rpy_degrees(R: np.ndarray) -> tuple[float, float, float]:
    """Extract roll, pitch, yaw for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return tuple(float(math.degrees(a)) for a in (roll, pitch, yaw))


def read_colmap_images(path: Path) -> dict[str, dict]:
    """Read COLMAP images.txt; retain each registered image's world-to-camera pose."""
    records: dict[str, dict] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        fields = line.split(maxsplit=9)
        if len(fields) < 10:
            raise ValueError(f"No se pudo interpretar la línea {i + 1} de {path}: {line}")

        try:
            image_id = int(fields[0])
            qw, qx, qy, qz = map(float, fields[1:5])
            tx, ty, tz = map(float, fields[5:8])
            camera_id = int(fields[8])
        except ValueError as exc:
            raise ValueError(f"Cabecera de imagen inválida en línea {i + 1}: {line}") from exc

        name = fields[9]
        if name in records:
            raise ValueError(f"Nombre de imagen duplicado en images.txt: {name}")

        R_wc = quaternion_to_rotation(qw, qx, qy, qz)
        t_wc = np.array([tx, ty, tz], dtype=float)
        center = -R_wc.T @ t_wc

        records[name] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "R_world_to_camera": R_wc,
            "center_colmap": center,
        }
        # The next line contains 2D observations (possibly empty).
        i += 2

    if not records:
        raise ValueError(f"No se encontraron imágenes registradas en {path}")
    return records


def read_references(path: Path) -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"Formato inválido en {path}, línea {line_number}. "
                "Se esperaba: nombre_imagen x y z"
            )
        name = fields[0]
        if name in refs:
            raise ValueError(f"Referencia duplicada: {name}")
        refs[name] = np.array([float(v) for v in fields[1:]], dtype=float)
    if not refs:
        raise ValueError(f"No se encontraron referencias en {path}")
    return refs


def estimate_similarity_umeyama(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate target ~= scale * R @ source + t, enforcing det(R)=+1."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Las matrices de puntos deben tener la misma forma Nx3.")
    if source.shape[0] < 3:
        raise ValueError("Se necesitan al menos 3 correspondencias.")

    mean_src = source.mean(axis=0)
    mean_dst = target.mean(axis=0)
    src_centered = source - mean_src
    dst_centered = target - mean_dst

    covariance = (dst_centered.T @ src_centered) / source.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
    R = U @ correction @ Vt

    variance_source = float(np.sum(src_centered ** 2) / source.shape[0])
    if variance_source < 1e-15:
        raise ValueError("Las posiciones COLMAP de referencia no tienen dispersión suficiente.")

    scale = float(np.sum(singular_values * np.diag(correction)) / variance_source)
    translation = mean_dst - scale * (R @ mean_src)

    if np.linalg.det(R) < 0.999999:
        raise RuntimeError("La rotación estimada no es propia (determinante distinto de +1).")
    return scale, R, translation


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pose_row(name: str, image_id: int, position: np.ndarray, R_world_body: np.ndarray) -> dict:
    qw, qx, qy, qz = rotation_to_quaternion(R_world_body)
    roll, pitch, yaw = rotation_to_rpy_degrees(R_world_body)
    return {
        "image_name": name,
        "image_id": image_id,
        "x_m": position[0],
        "y_m": position[1],
        "z_m": position[2],
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alinea las poses COLMAP al sistema físico FLU y valida referencias."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.images.is_file():
        raise FileNotFoundError(f"No existe el archivo: {args.images}")
    if not args.references.is_file():
        raise FileNotFoundError(f"No existe el archivo: {args.references}")

    images = read_colmap_images(args.images)
    references = read_references(args.references)

    common_names = [name for name in references if name in images]
    missing_names = [name for name in references if name not in images]
    if len(common_names) < 3:
        raise RuntimeError(
            f"Solo se encontraron {len(common_names)} referencias en COLMAP; se necesitan al menos 3."
        )

    source = np.vstack([images[name]["center_colmap"] for name in common_names])
    target = np.vstack([references[name] for name in common_names])

    scale, R_align, translation = estimate_similarity_umeyama(source, target)
    aligned_references = (scale * (R_align @ source.T)).T + translation
    residual_vectors = aligned_references - target
    errors = np.linalg.norm(residual_vectors, axis=1)

    args.output.mkdir(parents=True, exist_ok=True)

    drone_rows = []
    for name, record in images.items():
        center_colmap = record["center_colmap"]
        position_world = scale * (R_align @ center_colmap) + translation

        # COLMAP stores world-to-camera. Invert it to obtain camera-to-world.
        R_camera_world_colmap = record["R_world_to_camera"].T
        R_camera_world_physical = R_align @ R_camera_world_colmap

        # Camera optical axes -> body FLU axes:
        # v_body = R_BODY_FROM_CAMERA @ v_camera.
        # Therefore body-to-world = camera-to-world @ (camera-to-body).
        R_world_body = R_camera_world_physical @ R_BODY_FROM_CAMERA.T

        drone_rows.append(
            pose_row(name, record["image_id"], position_world, R_world_body)
        )

    pose_fields = [
        "image_name", "image_id", "x_m", "y_m", "z_m",
        "qx", "qy", "qz", "qw", "roll_deg", "pitch_deg", "yaw_deg",
    ]
    write_csv(args.output / "poses_drone_FLU.csv", pose_fields, drone_rows)

    np.savetxt(
        args.output / "similarity_transform_colmap_to_FLU.txt",
        np.vstack([np.column_stack([scale * R_align, translation]), [0, 0, 0, 1]]),
        fmt="%.12f",
        header="Transformación homogénea: p_FLU = scale * R_align * p_COLMAP + translation",
    )

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median = float(np.median(errors))
    max_error = float(np.max(errors))
    max_index = int(np.argmax(errors))
    rms_axis = np.sqrt(np.mean(residual_vectors ** 2, axis=0))

    summary_lines = [
        "ALINEACIÓN COLMAP -> SISTEMA FÍSICO FLU",
        "======================================",
        f"Archivo de imágenes: {args.images}",
        f"Archivo de referencias: {args.references}",
        f"Imágenes registradas: {len(images)}",
        f"Referencias encontradas: {len(common_names)}/{len(references)}",
        f"Referencias faltantes: {', '.join(missing_names) if missing_names else 'ninguna'}",
        "",
        "Transformación estimada con Umeyama/Kabsch (rotación propia, det=+1):",
        f"Escala total (m/unidad COLMAP): {scale:.12f}",
        f"Traslación (m): [{translation[0]:.9f}, {translation[1]:.9f}, {translation[2]:.9f}]",
        f"Determinante de R: {np.linalg.det(R_align):.12f}",
        "Matriz R_align:",
        np.array2string(R_align, precision=9, suppress_small=True),
        "",
        "Errores de las referencias (ajuste calculado con estas mismas referencias):",
        f"MAE 3D (m): {mae:.6f}",
        f"RMSE 3D (m): {rmse:.6f}",
        f"Mediana 3D (m): {median:.6f}",
        f"Error máximo 3D (m): {max_error:.6f} ({common_names[max_index]})",
        f"RMSE por eje X/Y/Z (m): [{rms_axis[0]:.6f}, {rms_axis[1]:.6f}, {rms_axis[2]:.6f}]",
        "",
        "Orientación:",
        "COLMAP: cuaternión world-to-camera en images.txt.",
        "poses_drone_FLU.csv: orientación del cuerpo FLU en el mundo físico.",
        "Euler: RPY en grados, convención ZYX (R = Rz(yaw) Ry(pitch) Rx(roll)).",
        "",
        "Supuesto de posición: el centro de cámara se usa como posición del dron.",
        "No se aplica traslación cámara-cuerpo; si el centro de cámara no coincide con",
        "el origen físico del dron, será necesario calibrar e incorporar ese offset.",
        "",
        "Nota: los errores son residuales sobre las mismas referencias usadas para",
        "estimar la transformación; se recomienda validar con puntos independientes.",
    ]
    (args.output / "alignment_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("\nALINEACIÓN Y EXPORTACIÓN COMPLETADAS")
    print(f"Imágenes registradas: {len(images)}")
    print(f"Referencias utilizadas: {len(common_names)}/{len(references)}")
    print(f"Escala total: {scale:.9f} m/unidad COLMAP")
    print(f"MAE 3D: {mae:.4f} m")
    print(f"RMSE 3D: {rmse:.4f} m")
    print(f"Mediana: {median:.4f} m")
    print(f"Error máximo: {max_error:.4f} m ({common_names[max_index]})")
    print(f"det(R_align): {np.linalg.det(R_align):.9f}")
    print(f"Resultados: {args.output.resolve()}")


if __name__ == "__main__":
    main()