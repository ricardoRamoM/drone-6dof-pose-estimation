# export_colmap_poses.py
# Leer las poses de COLMAP y generar poses_colmap.csv.

"""Exporta centros, orientaciones y ángulos RPY desde el images.txt de COLMAP.

Uso desde la raíz del repositorio:
    python src/post_colmap/export_colmap_poses.py
    python src/post_colmap/export_colmap_poses.py --input results_colmap/v1_0/sparse/1_txt/images.txt
    python src/post_colmap/export_colmap_poses.py --input .../images.txt --output dataset/v1_0/poses_colmap.csv

La conversión de posición es:
    C = -R^T t
porque COLMAP expresa:  X_c = R X_w + t.

Convención RPY (grados):
    R_c2w = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    roll  alrededor de X, pitch alrededor de Y, yaw alrededor de Z.
"""

from pathlib import Path
import argparse
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "results_colmap/v1_0/sparse/1_txt/images.txt"
DEFAULT_OUTPUT = ROOT / "dataset/v1_0/poses_colmap.csv"


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Matriz R de COLMAP para el cuaternión [qw, qx, qy, qz]."""
    q = np.asarray([qw, qx, qy, qz], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Cuaternión COLMAP nulo o no finito.")
    q /= norm
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def rotation_matrix_to_euler_rpy(R):
    """Devuelve (roll, pitch, yaw) en radianes para R = Rz(yaw) Ry(pitch) Rx(roll)."""
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) < 1.0 - 1e-7:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Caso de bloqueo de cardán (pitch = ±90°)
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def export_poses(input_file, output_file):
    poses = []
    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.is_file():
        raise FileNotFoundError(f"No existe el archivo COLMAP: {input_file}")

    with input_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            if len(fields) < 10:
                continue
            try:
                int(fields[0])
                qw, qx, qy, qz = map(float, fields[1:5])
                tx, ty, tz = map(float, fields[5:8])
                int(fields[8])
            except ValueError:
                # Línea de puntos 2D que sigue a cada imagen.
                continue

            image_name = " ".join(fields[9:])
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                continue

            # R: mundo -> cámara
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            t = np.asarray([tx, ty, tz], dtype=float)

            # Centro de cámara en mundo
            C = -R.T @ t

            # R_c2w (cámara -> mundo). Al ser R ortonormal, R_c2w = R^T.
            R_c2w = R.T
            roll, pitch, yaw = rotation_matrix_to_euler_rpy(R_c2w)

            poses.append({
                "image_name": image_name,
                "x": float(C[0]), "y": float(C[1]), "z": float(C[2]),
                # Cuaternión cámara-a-mundo (conjugado del de COLMAP)
                "qx": float(-qx), "qy": float(-qy), "qz": float(-qz),
                "qw": float(qw),
                # Ángulos RPY en grados (convención R = Rz Ry Rx)
                "roll_deg":  float(np.degrees(roll)),
                "pitch_deg": float(np.degrees(pitch)),
                "yaw_deg":   float(np.degrees(yaw)),
            })

    if not poses:
        raise ValueError(f"No se encontraron poses de imágenes en {input_file}")

    names = [pose["image_name"] for pose in poses]
    if len(names) != len(set(names)):
        raise ValueError("COLMAP produjo nombres de imagen duplicados; revisa images.txt.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "image_name",
        "x", "y", "z",
        "qx", "qy", "qz", "qw",
        "roll_deg", "pitch_deg", "yaw_deg",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(poses)

    print(f"Archivo generado: {output_file}")
    print(f"Poses exportadas: {len(poses)}")
    print("Primeras 5 poses (pos + quat c2w + RPY en grados):")
    for pose in poses[:5]:
        print(pose)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Ruta al images.txt exportado por COLMAP.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="CSV de salida.")
    args = parser.parse_args()
    export_poses(args.input, args.output)


if __name__ == "__main__":
    main()