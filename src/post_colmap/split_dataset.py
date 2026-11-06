#!/usr/bin/env python3
# src/post_colmap/split_dataset.py
"""Divide train/val por vídeo y aplica overrides de manual_review.txt.

Entrada:
    results_colmap/<v>/alignment/pose_validation_all.csv   (maestro)
    results_colmap/<v>/alignment/manual_review.txt         (opcional, editable)

Salida:
    dataset/<v>/poses_split.csv  (coma-separador, punto-decimal,
                                   solo columnas necesarias)

python src/post_colmap/split_dataset.py

"""

from pathlib import Path
import argparse
import csv
import re
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset"
RESULTS_ROOT = ROOT / "results_colmap"

DEFAULT_DATASET_VERSION = "v2_0"

VIDEO_SPLITS = {
    "train": {
        "bebop_002_train",
        "bebop_003_train",
        "bebop_004_train",
        "bebop_005_train",
        "ref_frame",
    },
    "val": {"bebop_001_val"},
}

VALID_STATUS = "valid"
ALLOWED_STATUSES = {"valid", "suspicious", "invalid"}

# Sólo para sugerir el nombre del vídeo en mensajes de error.
# El matching real lo hace _build_video_patterns().
VIDEO_FRAME_PATTERN = re.compile(
    r"^(?P<video>.+?)(?:_frame)?_?\d+\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)

RENAME = {"x_m": "x", "y_m": "y", "z_m": "z"}
POSE_COLS = ["x", "y", "z", "qx", "qy", "qz", "qw"]

OUTPUT_COLUMNS = [
    "image_name", "image_id",
    "x", "y", "z",
    "qx", "qy", "qz", "qw",
    "video_name", "frame_index", "split",
]
NUMERIC_FIELDS = set(POSE_COLS) | {"image_id", "frame_index"}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _detect_dialect(path: Path):
    sample = path.read_text(encoding="utf-8")[:65536]
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;")
    except csv.Error:
        return csv.excel


def _fix_decimal_comma(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    if not v:
        return v
    try:
        float(v)
        return v
    except ValueError:
        pass
    cand = v.replace(",", ".")
    try:
        float(cand)
        return cand
    except ValueError:
        return v


def load_review_overrides(path: Path) -> dict[str, str]:
    """Lee manual_review.txt: <image_name><TAB><status>, ignora #comentarios."""
    if not path.is_file():
        return {}
    overrides: dict[str, str] = {}
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace("\t", " ").split()
        if len(parts) < 2:
            raise ValueError(
                f"{path}:{lineno}: se esperaba '<image_name> <status>'"
            )
        name, status = parts[0], parts[1].lower()
        if status not in ALLOWED_STATUSES:
            raise ValueError(
                f"{path}:{lineno}: status '{status}' no válido "
                f"(usa {sorted(ALLOWED_STATUSES)})"
            )
        if name in overrides:
            raise ValueError(f"{path}: {name} aparece duplicado")
        overrides[name] = status
    return overrides


def _compile_video_patterns() -> dict[tuple[str, str], re.Pattern]:
    """Un regex precompilado por vídeo: acepta `<v>_frame_<i>` y `<v>_<i>`."""
    return {
        (split_name, video): re.compile(
            rf"{re.escape(video)}(?:_frame)?_\d+\.(?:jpg|jpeg|png)",
            re.IGNORECASE,
        )
        for split_name, videos in VIDEO_SPLITS.items()
        for video in videos
    }


_VIDEO_PATTERNS = _compile_video_patterns()

def _build_video_patterns(strict: bool) -> list[tuple[tuple[str, str], re.Pattern]]:
    """Un regex por vídeo. En modo laxo acepta `<v>_frame_1` y `<v>_frame1`;
    en modo estricto exige siempre `_` delante del índice."""
    sep = "_" if strict else "_?"
    patterns = [
        (
            (split_name, video),
            re.compile(
                rf"{re.escape(video)}(?:_frame)?{sep}\d+\.(?:jpg|jpeg|png)",
                re.IGNORECASE,
            ),
        )
        for split_name, videos in VIDEO_SPLITS.items()
        for video in videos
    ]
    # Más largo primero: evita que un vídeo prefijo de otro (p.ej. "ref"
    # vs "ref_frame") se coma el match del más específico.
    patterns.sort(key=lambda item: len(item[0][1]), reverse=True)
    return patterns

def get_split(image_name: str, patterns):
    filename = Path(image_name).name
    for (split_name, video), pattern in patterns:
        if pattern.fullmatch(filename):
            return split_name, video
    match = VIDEO_FRAME_PATTERN.match(filename)
    return None, (match.group("video") if match else image_name)


# --------------------------------------------------------------------------- #
# Núcleo
# --------------------------------------------------------------------------- #
def split_dataset(
    input_file: Path,
    output_file: Path,
    review_file: Path | None = None,
    strict: bool = False,
) -> None:
    if not input_file.is_file():
        raise FileNotFoundError(f"No existe el CSV de poses: {input_file}")

    patterns = _build_video_patterns(strict)

    dialect = _detect_dialect(input_file)
    with input_file.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, dialect=dialect)
        fieldnames = [
            (f or "").strip().lstrip("\ufeff")
            for f in (reader.fieldnames or [])
        ]
        rows = list(reader)

    if not rows or not fieldnames:
        raise ValueError("El CSV está vacío o no contiene encabezados.")

    rows = [
        {(k or "").strip().lstrip("\ufeff"): v for k, v in r.items()}
        for r in rows
    ]

    have_canonical = set(POSE_COLS) <= set(fieldnames)
    have_raw = {"x_m", "y_m", "z_m"} <= set(fieldnames)
    if not (have_canonical or have_raw):
        raise ValueError("Faltan columnas de posición (x_m/y_m/z_m o x/y/z).")

    missing = {"image_name", "status"} - set(fieldnames)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    image_names = [r["image_name"] for r in rows]
    if len(image_names) != len(set(image_names)):
        raise ValueError("El CSV contiene nombres de imagen duplicados.")

    # ------------------ Overrides de manual_review.txt ------------------ #
    review_overrides = load_review_overrides(review_file) if review_file else {}
    known_names = set(image_names)
    unknown = set(review_overrides) - known_names
    if unknown:
        raise ValueError(
            f"{review_file}: nombres que no están en el CSV: "
            f"{sorted(unknown)[:10]}"
        )

    n_overridden = 0
    for row in rows:
        new_status = review_overrides.get(row["image_name"])
        if new_status and row["status"] != new_status:
            row["status"] = new_status
            n_overridden += 1

    # ------------------ Filtrado por status == valid ------------------ #
    status_counts = Counter(
        (r.get("status") or "").strip().lower() for r in rows
    )
    excluded_status_counts = Counter(
        (r.get("status") or "").strip().lower()
        for r in rows
        if (r.get("status") or "").strip().lower() != VALID_STATUS
    )
    valid_rows = [
        r for r in rows
        if (r.get("status") or "").strip().lower() == VALID_STATUS
    ]

    counts = Counter()
    groups = defaultdict(set)
    output_rows = []
    unassigned = set()

    for row in valid_rows:
        split_name, group_name = get_split(row["image_name"], patterns)
        if split_name is None:
            unassigned.add(group_name or row["image_name"])
            continue

        canonical = {}
        for key, value in row.items():
            col = RENAME.get(key, key)
            canonical[col] = (
                _fix_decimal_comma(value) if col in NUMERIC_FIELDS
                else (value or "").strip()
            )

        clean_row = {c: canonical.get(c, "") for c in OUTPUT_COLUMNS}
        clean_row["split"] = split_name
        output_rows.append(clean_row)
        counts[split_name] += 1
        groups[split_name].add(group_name)

    if unassigned:
        raise ValueError(
            "Hay imágenes válidas sin asignación. Revisa nombres y VIDEO_SPLITS:\n  - "
            + "\n  - ".join(sorted(unassigned))
        )
    if not output_rows:
        raise ValueError("No quedaron imágenes válidas para train/val.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    train_groups = groups["train"] - {"reference_images"}
    overlap = train_groups & groups["val"]
    if overlap:
        raise RuntimeError(f"Fuga entre train y val: {sorted(overlap)}")

    # ------------------ Reporte ------------------ #
    print(f"Archivo generado: {output_file}")
    print(f"Poses en CSV original: {len(rows)}")
    print(f"Estados finales: {dict(status_counts)}")
    modo = "estricto" if strict else "laxo (acepta sin '_' antes del índice)"
    print(f"  - Modo de nombres: {modo}")
    if review_overrides:
        print(f"  - Overrides aplicados desde {review_file}: {n_overridden}")
    print(f"Excluidas por no ser 'valid': {sum(excluded_status_counts.values())}")
    if excluded_status_counts:
        print(f"  Detalle: {dict(excluded_status_counts)}")
    print(f"Imágenes incluidas: {len(output_rows)}")
    for split_name in ("train", "val"):
        print(f"\n{split_name}: {counts[split_name]} imágenes")
        for group_name in sorted(groups[split_name]):
            print(f"  - {group_name}")
    print("\nComprobación completada: ningún video aparece en más de un conjunto.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--review", type=Path, default=None,
        help="TXT con overrides. Por defecto: "
             "results_colmap/<v>/alignment/manual_review.txt",
    )
    parser.add_argument(
        "--strict-names", action="store_true",
        help="Exige '_' antes del índice (p.ej. ref_frame_001.jpg). "
            "Por defecto se acepta también ref_frame001.jpg.",
    )
    args = parser.parse_args()

    results_dir = RESULTS_ROOT / args.version
    dataset_dir = DATASET_ROOT / args.version

    if args.input is None:
        args.input = results_dir / "alignment" / "pose_validation_all.csv"
    if args.output is None:
        args.output = dataset_dir / "poses_split.csv"
    if args.review is None:
        args.review = results_dir / "alignment" / "manual_review.txt"

    if not args.input.is_file():
        parser.error(f"No existe el CSV de entrada: {args.input}")

    split_dataset(args.input, args.output, args.review, strict=args.strict_names)


if __name__ == "__main__":
    main()