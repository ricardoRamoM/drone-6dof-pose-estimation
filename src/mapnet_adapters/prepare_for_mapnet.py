#!/usr/bin/env python3
"""
prepare_for_mapnet.py

Convierte el dataset Bebop al formato esperado por geomapnet.

Entradas:
    dataset/v1_0_v2_0/frames/<seq>/<img>
    results_colmap/v2_0/alignment/poses_drone_FLU.csv

Salida:
    data/bebop/bebop/<seq>/<img>  + poses.txt
    data/bebop/bebop/{TrainSplit,TestSplit}.txt
    data/bebop/bebop/stats.txt
    data/bebop/bebop/pose_stats.txt
"""
import os
import sys
import glob
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path("/data/drone-6dof-pose-estimation")

# ─── CONFIG (ajusta si cambia algo) ───
FRAMES_ROOT = REPO / "dataset/v1_0_v2_0/frames"
POSES_CSV   = REPO / "results_colmap/v2_0/alignment/poses_drone_FLU.csv"
OUT_ROOT    = REPO / "data/bebop/bebop"

# Orden de las columnas del CSV (AUTO = autodetecta por nombre de header)
CSV_IMG_COL  = "auto"     # "auto" | "image" | "filename" | "name" | ...
CSV_SEQ_COL  = "auto"     # "auto" | "sequence" | "seq" | None
CSV_XYZQ_COLS = "auto"    # "auto" | (x_col, y_col, z_col, qx_col, qy_col, qz_col, qw_col)

TRAIN_SEQS = ["bebop_002_train", "bebop_003_train",
              "bebop_004_train", "bebop_005_train"]
TEST_SEQS  = ["bebop_001_val"]

USE_SYMLINK = False   # True para no duplicar ~1189 PNGs
# ──────────────────────────────────────


def find_col(df, candidates):
    for c in candidates:
        for col in df.columns:
            if col.strip().lower() == c.lower():
                return col
    return None


def load_poses_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(f"CSV cargado: {csv_path.name}  ({len(df)} filas, columnas: {list(df.columns)})")

    # Columna de imagen
    if CSV_IMG_COL == "auto":
        img_col = find_col(df, ["image", "filename", "name", "file", "img",
                                "image_name", "frame", "path"])
        if img_col is None:
            raise SystemExit(f"No encuentro columna de imagen en {list(df.columns)}")
    else:
        img_col = CSV_IMG_COL

    # Columna de secuencia (opcional)
    if CSV_SEQ_COL == "auto":
        seq_col = find_col(df, ["sequence", "seq", "split", "sequence_name"])
    else:
        seq_col = CSV_SEQ_COL

    # Columnas x y z qx qy qz qw
    if CSV_XYZQ_COLS == "auto":
        cols = [find_col(df, [c]) for c in
                ["x", "y", "z", "qx", "qy", "qz", "qw"]]
        if any(c is None for c in cols):
            # Prueba con wxyz invertido
            alt = [find_col(df, [c]) for c in
                   ["x", "y", "z", "qw", "qx", "qy", "qz"]]
            if any(c is None for c in alt):
                raise SystemExit(f"No encuentro columnas de pose en {list(df.columns)}")
            raise SystemExit(
                "Columnas de cuaternión en orden (qw,qx,qy,qz). "
                "Reordena el CSV o ajusta este script.")
        xyzq = cols
    else:
        xyzq = list(CSV_XYZQ_COLS)

    rows = []
    for _, row in df.iterrows():
        img_field = str(row[img_col])
        # Ruta puede ser "seq/img.jpg" o solo "img.jpg"
        if "/" in img_field or os.sep in img_field:
            seq, fname = img_field.replace(os.sep, "/").split("/", 1)
        else:
            seq = row[seq_col] if seq_col is not None else None
            fname = img_field
        x, y, z, qx, qy, qz, qw = (float(row[c]) for c in xyzq)
        rows.append((seq, fname, x, y, z, qx, qy, qz, qw))

    print(f"  → imagen: {img_col!r}  | seq: {seq_col!r}  | pose: {xyzq}")
    return rows


def locate_image(seq, fname):
    """Encuentra la imagen en el dataset. Prueba varias extensiones."""
    if seq is not None:
        base = FRAMES_ROOT / seq / fname
        if base.is_file():
            return base
        stem = Path(fname).stem
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            cand = FRAMES_ROOT / seq / (stem + ext)
            if cand.is_file():
                return cand
        return None
    # Sin seq: buscar en todas las carpetas
    stem = Path(fname).stem
    for d in FRAMES_ROOT.iterdir():
        if not d.is_dir() or d.name in ("ref_frame", "frames_all"):
            continue
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            cand = d / (stem + ext)
            if cand.is_file():
                return cand
    return None


def main():
    if not POSES_CSV.is_file():
        sys.exit(f"Falta {POSES_CSV}")
    if not FRAMES_ROOT.is_dir():
        sys.exit(f"Falta {FRAMES_ROOT}")

    rows = load_poses_csv(POSES_CSV)

    # Agrupar por secuencia
    by_seq = {}
    missing = 0
    for (seq, fname, x, y, z, qx, qy, qz, qw) in rows:
        img = locate_image(seq, fname)
        if img is None:
            missing += 1
            continue
        seq_name = seq or img.parent.name
        by_seq.setdefault(seq_name, []).append(
            (img, x, y, z, qx, qy, qz, qw))

    print(f"Localizadas {sum(len(v) for v in by_seq.values())} imágenes, "
          f"{missing} no encontradas")

    # Solo secuencias que nos interesan
    all_seqs = set(by_seq)
    keep = set(TRAIN_SEQS) | set(TEST_SEQS)
    extra = all_seqs - keep
    if extra:
        print(f"⚠  Secuencias en CSV no incluidas: {sorted(extra)}")

    # Escribir por secuencia
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for seq_name in TRAIN_SEQS + TEST_SEQS:
        if seq_name not in by_seq:
            print(f"⚠  {seq_name} no aparece en CSV, salto")
            continue
        entries = sorted(by_seq[seq_name], key=lambda e: e[0].name)
        seq_out = OUT_ROOT / seq_name
        seq_out.mkdir(parents=True, exist_ok=True)
        with open(seq_out / "poses.txt", "w") as f:
            f.write("# filename x y z qx qy qz qw\n")
            for (img, x, y, z, qx, qy, qz, qw) in entries:
                dst = seq_out / img.name
                if not dst.exists():
                    if USE_SYMLINK:
                        os.symlink(img, dst)
                    else:
                        shutil.copy2(img, dst)
                f.write(f"{img.name} {x:.9f} {y:.9f} {z:.9f} "
                        f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
        print(f"[{seq_name}] {len(entries)} frames")

    # Splits
    with open(OUT_ROOT / "TrainSplit.txt", "w") as f:
        f.write("\n".join(TRAIN_SEQS) + "\n")
    with open(OUT_ROOT / "TestSplit.txt", "w") as f:
        f.write("\n".join(TEST_SEQS) + "\n")
    print(f"TrainSplit: {TRAIN_SEQS}")
    print(f"TestSplit : {TEST_SEQS}")

    # stats.txt (RGB) en 2 pasadas
    print("Calculando stats RGB...")
    means = np.zeros(3); n = 0
    for seq in TRAIN_SEQS + TEST_SEQS:
        for p in glob.glob(str(OUT_ROOT / seq / "*.color.png")) + \
                 glob.glob(str(OUT_ROOT / seq / "*.jpg")):
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
            flat = arr.reshape(-1, 3)
            means += flat.sum(0); n += flat.shape[0]
    means /= max(n, 1)
    sq = np.zeros(3)
    for seq in TRAIN_SEQS + TEST_SEQS:
        for p in glob.glob(str(OUT_ROOT / seq / "*.color.png")) + \
                 glob.glob(str(OUT_ROOT / seq / "*.jpg")):
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
            flat = arr.reshape(-1, 3)
            sq += ((flat - means) ** 2).sum(0)
    stds = np.sqrt(sq / max(n, 1))
    with open(OUT_ROOT / "stats.txt", "w") as f:
        f.write(" ".join(f"{v:.9e}" for v in means) + "\n")
        f.write(" ".join(f"{v:.9e}" for v in stds) + "\n")
    print(f"stats.txt: mean={means.round(4)}, std={stds.round(4)}")

    # pose_stats.txt (placeholder 0/1, igual que heads)
    with open(OUT_ROOT / "pose_stats.txt", "w") as f:
        f.write("0.0 0.0 0.0\n1.0 1.0 1.0\n")
    print("pose_stats.txt: placeholder 0/1 (consistente con el preentrenado heads)")


if __name__ == "__main__":
    main()