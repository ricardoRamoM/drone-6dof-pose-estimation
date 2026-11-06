# split_dataset.py
# Dividir el conjunto de datos en entrenamiento y validación.
# Asignar videos completos a entrenamiento, validación y prueba.

"""Asigna videos completos a train/val/test y genera poses_colmap_split.csv.

La configuración inicial reproduce la división utilizada en el proyecto Bebop actual.
Para otro escenario, edita VIDEO_SPLITS antes de ejecutar el script.

Uso:
    python src/split_dataset.py
    python src/split_dataset.py --input data/poses_colmap.csv --output data/poses_colmap_split.csv
"""
from pathlib import Path
import argparse
import csv
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data/poses_colmap.csv"
DEFAULT_OUTPUT = ROOT / "data/poses_colmap_split.csv"

# Esta configuración corresponde ÚNICAMENTE a los videos del dataset actual.
VIDEO_SPLITS = {
    "train": {
        "bebop_001_desde_piso",
        "bebop_002_avance_bajo",
        "bebop_003_elevacion",
        "bebop_004_avance_elevacion",
        "bebop_005_pared_1",
        "bebop_007_avance_giro",
        "bebop_009_esquina_opuesta",
        "bebop_012_recorrido_completo",
    },
    "val": {
        "bebop_008_diagonal",
        "bebop_010_recorrido_lateral",
    },
    "test": {
        "bebop_006_pared_2",
        "bebop_011_recorrido_circular",
    },
}


def split_dataset(input_file, output_file):
    input_file, output_file = Path(input_file), Path(output_file)
    if not input_file.is_file():
        raise FileNotFoundError(f"No existe el CSV de poses: {input_file}")

    video_to_split = {}
    for split_name, videos in VIDEO_SPLITS.items():
        for video in videos:
            if video in video_to_split:
                raise ValueError(f"Video asignado más de una vez: {video}")
            video_to_split[video] = split_name

    with input_file.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows or not fieldnames or "image_name" not in fieldnames:
        raise ValueError("El CSV está vacío o no contiene la columna image_name.")

    counts = Counter()
    videos_by_split = {name: set() for name in VIDEO_SPLITS}
    unassigned = set()
    duplicate_images = len(rows) != len({row["image_name"] for row in rows})

    for row in rows:
        # Eliminar split previo para evitar columnas duplicadas.
        image_name = row["image_name"]
        video_name = image_name.split("_frame_", 1)[0]
        split_name = video_to_split.get(video_name)
        if split_name is None:
            unassigned.add(video_name)
            continue
        row["split"] = split_name
        counts[split_name] += 1
        videos_by_split[split_name].add(video_name)

    if unassigned:
        raise ValueError(
            "Hay videos sin asignar en VIDEO_SPLITS: " + ", ".join(sorted(unassigned))
        )
    if duplicate_images:
        raise ValueError("El CSV contiene nombres de imagen duplicados.")

    output_fields = [column for column in fieldnames if column != "split"] + ["split"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    all_groups = list(videos_by_split.values())
    for i in range(len(all_groups)):
        for j in range(i + 1, len(all_groups)):
            overlap = all_groups[i] & all_groups[j]
            if overlap:
                raise RuntimeError(f"Fuga entre conjuntos: {sorted(overlap)}")

    print(f"Archivo generado: {output_file}")
    print(f"Total de poses: {len(rows)}")
    for split_name in ("train", "val", "test"):
        print(f"\n{split_name}: {counts[split_name]} imágenes, {len(videos_by_split[split_name])} videos")
        for video in sorted(videos_by_split[split_name]):
            print(f"  - {video}")
    print("\nComprobación completada: ningún video aparece en más de un conjunto.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    split_dataset(args.input, args.output)


if __name__ == "__main__":
    main()