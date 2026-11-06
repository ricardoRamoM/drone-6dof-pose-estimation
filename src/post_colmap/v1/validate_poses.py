# validate_poses.py	
# Comprobar que el CSV tenga poses válidas y que correspondan a imágenes existentes.
"""Valida estructura, imágenes, valores y cuaterniones de un CSV de poses.

Uso:
    python src/validate_poses.py
    python src/validate_poses.py --csv data/poses_colmap.csv --images data/frames_colmap_all
"""

from pathlib import Path
import argparse
import csv
import math
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data/poses_colmap.csv"
DEFAULT_IMAGES = ROOT / "data/frames_colmap_all"
REQUIRED = ["image_name", "x", "y", "z", "qx", "qy", "qz", "qw"]


def video_name_from_image(image_name):
    """Obtiene el prefijo del video a partir de nombres ..._frame_00001.jpg."""
    marker = "_frame_"
    return image_name.split(marker, 1)[0] if marker in image_name else "(formato no reconocido)"


def validate(csv_path, image_dir):
    csv_path, image_dir = Path(csv_path), Path(image_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {image_dir}")

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = list(reader)

    print("=== 1. Cantidad y estructura ===")
    print(f"Poses en CSV: {len(rows)}")
    print(f"Nombres de imagen únicos: {len(set(r.get('image_name', '') for r in rows))}")
    columns_ok = all(column in columns for column in REQUIRED)
    print(f"Columnas requeridas presentes: {columns_ok}")
    if not columns_ok:
        print("Columnas encontradas:", columns)
        print("Faltan:", sorted(set(REQUIRED) - set(columns)))

    print("\n=== 2. Comprobar imágenes ===")
    missing = [r.get("image_name", "") for r in rows
               if not (image_dir / r.get("image_name", "")).is_file()]
    print(f"Imágenes del CSV que no existen: {len(missing)}")
    if missing:
        print("Ejemplos:", missing[:10])

    print("\n=== 3. Comprobar valores numéricos y cuaterniones ===")
    invalid, bad_quaternions = [], []
    numeric_columns = REQUIRED[1:]
    for row in rows:
        name = row.get("image_name", "(sin nombre)")
        try:
            values = [float(row[key]) for key in numeric_columns]
            if not all(math.isfinite(value) for value in values):
                invalid.append(name)
                continue
            qx, qy, qz, qw = (float(row[key]) for key in ["qx", "qy", "qz", "qw"])
            norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if abs(norm - 1.0) > 1e-4:
                bad_quaternions.append((name, norm))
        except (ValueError, TypeError, KeyError):
            invalid.append(name)

    duplicate_names = len(rows) != len(set(r.get("image_name", "") for r in rows))
    print(f"Filas con valores inválidos: {len(invalid)}")
    print(f"Cuaterniones con norma distinta de 1: {len(bad_quaternions)}")
    print(f"Nombres duplicados: {duplicate_names}")
    if bad_quaternions:
        print("Ejemplos de normas incorrectas:", bad_quaternions[:5])

    print("\n=== 4. Distribución por video ===")
    counts = Counter(video_name_from_image(r.get("image_name", "")) for r in rows)
    for video, count in sorted(counts.items()):
        print(f"{video}: {count}")

    print("\n=== 5. Rangos de posición ===")
    if rows and not invalid:
        for axis in ["x", "y", "z"]:
            values = [float(r[axis]) for r in rows]
            print(f"{axis}: mínimo={min(values):.4f}, máximo={max(values):.4f}")

    passed = bool(rows) and columns_ok and not missing and not invalid and not bad_quaternions and not duplicate_names
    print("\n=== RESULTADO ===")
    print("VALIDACIÓN BÁSICA SUPERADA." if passed else "VALIDACIÓN FALLIDA: revisa los problemas anteriores.")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    args = parser.parse_args()
    if not validate(args.csv, args.images):
        raise SystemExit(1)


if __name__ == "__main__":
    main()