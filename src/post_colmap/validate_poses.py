#!/usr/bin/env python3
# validate_poses.py — Validate COLMAP-derived drone poses in FLU coordinates.
"""Validate COLMAP-derived drone poses expressed in the physical FLU frame.

Input:
    results_colmap/v2_0/alignment/poses_drone_FLU.csv

Outputs (written beside the input unless --output-dir is supplied):
    pose_validation_all.csv
    pose_validation_summary.txt

The validator does not modify the source CSV. Spatial limits and temporal
thresholds are diagnostic defaults; adjust them to the actual flight volume
and plausible Bebop motion before treating flags as ground truth.

python src/post_colmap/validate_poses.py
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    "results_colmap/v2_0/alignment/poses_drone_FLU.csv"
)

REQUIRED_COLUMNS = [
    "image_name", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"
]

# Broad diagnostic bounds for the measured room / flight area (metres).
DEFAULT_BOUNDS = {
    "x": (-0.3, 4.5),
    "y": (-3.0, 0.5),
    "z": (-0.3, 3.0),
}

DEFAULT_MAX_SPEED = 3.0
DEFAULT_MAX_ANGULAR_SPEED = 240.0
DEFAULT_QUATERNION_TOLERANCE = 1e-3

FRAME_PATTERN = re.compile(r"^(?P<video>.+)_frame_(?P<frame>\d+)\.(jpg|jpeg|png)$", re.I)

MANUAL_REVIEW_NAME = "manual_review.txt"
ALLOWED_STATUSES = {"valid", "suspicious", "invalid"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate aligned drone poses in FLU coordinates."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input pose CSV.")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Default: same directory as the input CSV."
    )
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frame extraction rate used in the filenames (default: 2).")
    parser.add_argument("--min-x", type=float, default=DEFAULT_BOUNDS["x"][0])
    parser.add_argument("--max-x", type=float, default=DEFAULT_BOUNDS["x"][1])
    parser.add_argument("--min-y", type=float, default=DEFAULT_BOUNDS["y"][0])
    parser.add_argument("--max-y", type=float, default=DEFAULT_BOUNDS["y"][1])
    parser.add_argument("--min-z", type=float, default=DEFAULT_BOUNDS["z"][0])
    parser.add_argument("--max-z", type=float, default=DEFAULT_BOUNDS["z"][1])
    parser.add_argument(
        "--strict-bounds", action="store_true",
        help="Treat positions outside the configured spatial bounds as INVALID. "
             "By default they are only SUSPICIOUS."
    )
    parser.add_argument(
        "--max-speed", type=float, default=DEFAULT_MAX_SPEED,
        help="Flag consecutive-pose translational speed above this value (m/s). "
             "Set <=0 to disable. Default: 3."
    )
    parser.add_argument(
        "--max-angular-speed", type=float, default=DEFAULT_MAX_ANGULAR_SPEED,
        help="Flag relative angular speed above this value (deg/s). "
             "Set <=0 to disable. Default: 240."
    )
    parser.add_argument(
        "--quaternion-tolerance", type=float, default=DEFAULT_QUATERNION_TOLERANCE,
        help="Allowed absolute deviation of quaternion norm from 1."
    )
    return parser.parse_args()


def quaternion_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Smallest relative rotation angle; q order is [qx, qy, qz, qw]."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def frame_metadata(image_name: str) -> tuple[str, float]:
    match = FRAME_PATTERN.match(str(image_name))
    if match:
        return match.group("video"), float(match.group("frame"))
    # Keep unparseable names together only if they share the same prefix.
    return str(image_name), float("nan")


def percentile_summary(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return "n/a"
    q = clean.quantile([0.50, 0.90, 0.95, 0.99])
    return (
        f"min={clean.min():.4f}, p50={q.loc[0.50]:.4f}, "
        f"p90={q.loc[0.90]:.4f}, p95={q.loc[0.95]:.4f}, "
        f"p99={q.loc[0.99]:.4f}, max={clean.max():.4f}"
    )

def load_review_overrides(path: Path) -> dict[str, str]:
    """Lee un manual_review.txt ya existente para preservar decisiones del usuario."""
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
        overrides[name] = status
    return overrides


def write_manual_review(path: Path, df: pd.DataFrame) -> None:
    """Genera/actualiza manual_review.txt a partir de las suspicious actuales.

    Si el fichero ya existe, preserva los status 'valid'/'invalid' que
    el usuario haya anotado. Las suspicious nuevas se añaden como
    'suspicious' para revisión.
    """
    existing = load_review_overrides(path)

    suspicious_df = df[df["status"] == "suspicious"].sort_values("image_name")

    # Detalle de cada suspicious, en bloque de comentarios al principio.
    detail_lines = ["# ---- Detalle ----"]
    for _, row in suspicious_df.iterrows():
        detail_lines.append(
            f"# {row['image_name']} | "
            f"XYZ=({row['x_m']:.3f}, {row['y_m']:.3f}, {row['z_m']:.3f}) | "
            f"{row['validation_reasons']}"
        )
    detail_lines.append("#")

    # Una línea por imagen, tabulada: <image_name>\t<status>
    entry_lines: list[str] = []
    for name in suspicious_df["image_name"]:
        status = existing.get(name, "suspicious")
        entry_lines.append(f"{name}\t{status}")

    header = [
        "# manual_review.txt",
        "# Cambia 'suspicious' por 'valid' o 'invalid' tras revisar cada imagen.",
        "# Las que dejes como 'suspicious' se excluirán del entrenamiento.",
        "# Formato: <image_name><TAB><status>",
        "# Este fichero se lee desde split_dataset.py.",
        "#",
    ]

    path.write_text(
        "\n".join(header + detail_lines + entry_lines) + "\n",
        encoding="utf-8",
    )

def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"No encuentro el CSV:\n  {input_path}\n"
            "Comprueba --input o ejecuta el script desde la raíz del repositorio."
        )
    if args.fps <= 0:
        raise ValueError("--fps debe ser mayor que cero.")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")

    df = df.copy()
    df["source_row"] = np.arange(len(df))
    df["video_name"] = df["image_name"].apply(lambda n: frame_metadata(n)[0])
    df["frame_index"] = df["image_name"].apply(lambda n: frame_metadata(n)[1])

    numeric_cols = ["x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Work in chronological order within each video. COLMAP image_id is NOT
    # used for ordering because it is not guaranteed to be chronological.
    df = df.sort_values(
        ["video_name", "frame_index", "source_row"],
        na_position="last"
    ).reset_index(drop=True)

    reasons: list[list[str]] = [[] for _ in range(len(df))]
    invalid = np.zeros(len(df), dtype=bool)
    suspicious = np.zeros(len(df), dtype=bool)

    def add_flag(i: int, reason: str, is_invalid: bool = False) -> None:
        if reason not in reasons[i]:
            reasons[i].append(reason)
        if is_invalid:
            invalid[i] = True
        else:
            suspicious[i] = True

    positions = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    quats = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
    finite_pose = np.isfinite(positions).all(axis=1) & np.isfinite(quats).all(axis=1)
    q_norm = np.linalg.norm(quats, axis=1)
    df["quaternion_norm"] = q_norm

    for i in range(len(df)):
        if not finite_pose[i]:
            add_flag(i, "non_finite_pose", True)
            continue
        if not np.isfinite(q_norm[i]) or q_norm[i] < 1e-8:
            add_flag(i, "invalid_quaternion_norm", True)
            continue
        if abs(q_norm[i] - 1.0) > args.quaternion_tolerance:
            add_flag(i, "quaternion_not_normalized", False)

        x, y, z = positions[i]
        outside = (
            x < args.min_x or x > args.max_x or
            y < args.min_y or y > args.max_y or
            z < args.min_z or z > args.max_z
        )
        if outside:
            reason = (
                f"outside_bounds(x={args.min_x:g}..{args.max_x:g},"
                f"y={args.min_y:g}..{args.max_y:g},"
                f"z={args.min_z:g}..{args.max_z:g})"
            )
            add_flag(i, reason, args.strict_bounds)

    # Temporal diagnostics: compare adjacent extracted frame indices within
    # each video. dt accounts for missing frame indices and extraction FPS.
    df["dt_s"] = np.nan
    df["step_distance_m"] = np.nan
    df["linear_speed_m_s"] = np.nan
    df["rotation_step_deg"] = np.nan
    df["angular_speed_deg_s"] = np.nan

    for _, group in df.groupby("video_name", sort=False):
        indices = group.index.to_list()
        for previous, current in zip(indices[:-1], indices[1:]):
            f0 = df.at[previous, "frame_index"]
            f1 = df.at[current, "frame_index"]
            if not (np.isfinite(f0) and np.isfinite(f1)) or f1 <= f0:
                continue
            if not (finite_pose[previous] and finite_pose[current]):
                continue
            dt = (f1 - f0) / args.fps
            distance = float(np.linalg.norm(positions[current] - positions[previous]))
            angle = quaternion_angle_deg(quats[previous], quats[current])
            speed = distance / dt
            angular_speed = angle / dt

            df.at[current, "dt_s"] = dt
            df.at[current, "step_distance_m"] = distance
            df.at[current, "linear_speed_m_s"] = speed
            df.at[current, "rotation_step_deg"] = angle
            df.at[current, "angular_speed_deg_s"] = angular_speed

            if args.max_speed > 0 and speed > args.max_speed:
                add_flag(current, f"high_linear_speed({speed:.2f}_m_s)")
                add_flag(previous, f"adjacent_to_high_linear_speed({speed:.2f}_m_s)")
            if args.max_angular_speed > 0 and angular_speed > args.max_angular_speed:
                add_flag(current, f"high_angular_speed({angular_speed:.1f}_deg_s)")
                add_flag(previous, f"adjacent_to_high_angular_speed({angular_speed:.1f}_deg_s)")

    # A row can be both invalid and suspicious; invalid takes precedence in
    # the status column, while all reasons remain in the master CSV.
    status = []
    for i in range(len(df)):
        if invalid[i]:
            status.append("invalid")
        elif suspicious[i]:
            status.append("suspicious")
        else:
            status.append("valid")
    df["status"] = status
    df["validation_reasons"] = ["; ".join(r) for r in reasons]

    output = df.drop(columns=["source_row"])
    output.to_csv(output_dir / "pose_validation_all.csv", index=False)

    lines = [
        "VALIDACIÓN DE POSES COLMAP — SISTEMA FLU",
        "=" * 58,
        f"Archivo de entrada: {input_path}",
        f"FPS usado para dt: {args.fps:g}",
        f"Total de poses: {len(df)}",
        f"Válidas: {int((df.status == 'valid').sum())}",
        f"Sospechosas: {int((df.status == 'suspicious').sum())}",
        f"Inválidas: {int((df.status == 'invalid').sum())}",
        "",
        "Límites espaciales (m):",
        f"  X: [{args.min_x}, {args.max_x}]",
        f"  Y: [{args.min_y}, {args.max_y}]",
        f"  Z: [{args.min_z}, {args.max_z}]",
        f"  Fuera de límites se considera {'INVALID' if args.strict_bounds else 'SUSPICIOUS'}.",
        "",
        "Umbrales temporales:",
        f"  Velocidad lineal máxima: {args.max_speed:g} m/s (<=0 desactiva)",
        f"  Velocidad angular máxima: {args.max_angular_speed:g} deg/s (<=0 desactiva)",
        "",
        "Distribuciones de pasos temporales (solo pares consecutivos comparables):",
        f"  Distancia: {percentile_summary(df['step_distance_m'])} m",
        f"  Velocidad lineal: {percentile_summary(df['linear_speed_m_s'])} m/s",
        f"  Giro relativo: {percentile_summary(df['rotation_step_deg'])} deg",
        f"  Velocidad angular: {percentile_summary(df['angular_speed_deg_s'])} deg/s",
        "",
        "Poses señaladas:",
    ]

    flagged = output[output["status"] != "valid"]
    if flagged.empty:
        lines.append("  Ninguna.")
    else:
        for _, row in flagged.iterrows():
            lines.append(
                f"  [{row['status'].upper()}] {row['image_name']} | "
                f"XYZ=({row['x_m']:.3f}, {row['y_m']:.3f}, {row['z_m']:.3f}) | "
                f"{row['validation_reasons']}"
            )

    lines += [
        "",
        "Notas:",
        "- El orden temporal se obtiene del número incluido en el nombre del frame,",
        "  no del image_id de COLMAP.",
        "- dt = diferencia de índices / FPS. Los frames faltantes se tienen en cuenta.",
        "- Un salto temporal marca ambas poses adyacentes como sospechosas; no las invalida.",
        "- Los límites espaciales son valores iniciales de diagnóstico, no una verdad universal.",
        "- Revisa visualmente las imágenes señaladas antes de excluirlas del entrenamiento.",
        "",
        "Archivos generados:",
        "  pose_validation_all.csv",
        "  pose_validation_summary.txt",
    ]
    (output_dir / "pose_validation_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    write_manual_review(output_dir / MANUAL_REVIEW_NAME, df)
    n_suspicious = int((df["status"] == "suspicious").sum())
    print(
        f"\nRevisión manual: {output_dir / MANUAL_REVIEW_NAME} "
        f"({n_suspicious} imágenes suspicious)"
    )

    print("\n".join(lines))
    print(f"\nResultados guardados en: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())