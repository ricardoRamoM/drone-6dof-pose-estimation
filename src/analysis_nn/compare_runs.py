"""Compara múltiples entrenamientos y genera material para asesoría.

Además de la comparación genérica, produce:
  - Extracción de los 5 números clave por run.
  - Tiempo total de entrenamiento por run (y por época) + figura dedicada.
  - Sanity check E08 vs E10 (deben coincidir hasta la época 80).
  - Sanity check E13/E14 (caídas de LR y early stopping).
  - Figuras curadas: pipeline, barrido, sobreajuste, control, scatter.
  - Diagnóstico overfit/underfit por run (rejilla small-multiples).
  - Comparación directa de 2 runs (par configurable en PAIR_DEFAULT).
  - Comparación de tiempos de ejecución (total y por época).

Uso:
    # comparar todos los runs bajo una raíz
    python -m src.analysis_nn.compare_runs outputs_nn/resnet18
    python -m src.analysis_nn.compare_runs outputs_nn/resnet18     --exclude E01 --exclude E02 --exclude E12

    # comparar solo 2-3 concretos (acepta E08, 08, run_08, nombre completo)
    python -m src.analysis_nn.compare_runs outputs_nn/resnet18 \
        --run E08 --run E10

    # solo los últimos 3
    python -m src.analysis_nn.compare_runs outputs_nn/resnet18 --last 3

    # todos menos uno
    python -m src.analysis_nn.compare_runs outputs_nn/resnet18 \
        --exclude E02

    # comparación directa de un par concreto (figura 13)
    python -m src.analysis_nn.compare_runs outputs_nn/resnet18 \
        --pair E06 E07
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Constantes compartidas — fuente única de verdad en __init__.py.
# NO las redefinas localmente: si las pisas, los tags y las líneas verdes
# dejarán de estar sincronizados.
from src.analysis_nn import (
    load_history, get_metrics, discover_runs,
    detect_patterns, find_latest_runs,
    OVERFIT_GAP_THRESHOLD, OVERFIT_RISE_PCT,
    UNDERFIT_MIN_LOSS, UNDERFIT_DROP_PCT,
    CONVERGED_SLOPE_EPS, CONVERGED_CV_EPS,
)


# ---------------------------------------------------------------------------
# Configuración por defecto
# ---------------------------------------------------------------------------

# Par por defecto para la figura 13 (comparación directa de 2 runs).
# Cambiar aquí o pasar `--pair A B` por CLI.
PAIR_DEFAULT = ("E08", "E14")  # E08 (wd=0) vs E14 (wd=1e-4)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _safe(values):
    if not values:
        return None
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def _len(arr) -> int:
    """Longitud segura de un ndarray | None (sin ambigüedad de truthiness)."""
    return 0 if arr is None else int(len(arr))


def _fmt(v, spec=".4f"):
    return f"{v:{spec}}" if isinstance(v, (int, float)) else "—"


def _fmt_time(seconds) -> str:
    """Formatea segundos como '1h 2m', '1m 23s' o '45.2s'."""
    if seconds is None:
        return "—"
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {seconds % 60:.0f}s"
    return f"{seconds:.1f}s"


def _epochs_array(arr):
    return np.arange(1, len(arr) + 1)


def _empty_panel(ax, name: str):
    ax.set_title(name, fontsize=10)
    ax.axis("off")
    ax.text(0.5, 0.5, "sin datos", ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color="gray")


def _extract_lr(metrics):
    """Busca la curva de learning rate en los keys más habituales."""
    for key in ("lr", "learning_rate", "LR", "Lr", "learning_rates"):
        arr = _safe(metrics.get(key))
        if arr is not None and not np.all(np.isnan(arr)):
            return arr, key
    return None, None


def _detect_lr_drops(lr: np.ndarray, min_ratio: float = 0.75):
    """Devuelve las épocas donde el LR cayó a <= min_ratio del valor anterior."""
    drops = []
    for i in range(1, len(lr)):
        prev, cur = lr[i - 1], lr[i]
        if np.isnan(prev) or np.isnan(cur) or prev == 0:
            continue
        if cur / prev <= min_ratio:
            drops.append(i + 1)  # época 1-based
    return drops


def _configured_epochs(history) -> int | None:
    """Intenta leer el número de épocas configuradas del history."""
    if not isinstance(history, dict):
        return None
    cfg = history.get("config", {}) or {}
    for src in (cfg, history):
        for key in ("epochs", "num_epochs", "max_epochs"):
            if isinstance(src, dict) and key in src:
                try:
                    return int(src[key])
                except (TypeError, ValueError):
                    pass
    return None


def _extract_training_time(run_dir: Path, history) -> float | None:
    """Extrae el tiempo total de entrenamiento (segundos) del history o de
    un run_info.json / metadata.json. Devuelve None si no se encuentra.

    Orden de búsqueda:
      1. history['config'], history['timing'], history['metrics'], history
      2. run_info.json / metadata.json / run_metadata.json en el run_dir
      3. Suma de history['epoch_times'] (o claves equivalentes) si existen
      4. Fallback: diferencia entre start_time y end_time (ISO 8601)
    """
    keys = (
        # ← NUEVO: clave real de tus run_info.json
        "total_duration_sec", "total_duration_seconds", "total_duration",
        # ya existentes
        "training_time_seconds", "training_time_s", "training_time",
        "elapsed_time_seconds", "elapsed_time_s", "elapsed_time",
        "duration_seconds", "duration_s", "duration",
        "total_time_seconds", "total_time_s", "total_time",
        "time_seconds", "time",
    )

    def _find(d: dict) -> float | None:
        for key in keys:
            if key in d:
                try:
                    val = float(d[key])
                    if val > 0:
                        return val
                except (TypeError, ValueError):
                    pass
        return None

    def _from_timestamps(d: dict) -> float | None:
        """start_time + end_time → segundos (por si falta el duration)."""
        if not isinstance(d, dict):
            return None
        s, e = d.get("start_time"), d.get("end_time")
        if not s or not e:
            return None
        try:
            from datetime import datetime
            # Acepta "2026-11-06T00:21:02" y variantes con zona/segundos
            t0 = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(str(e).replace("Z", "+00:00"))
            delta = (t1 - t0).total_seconds()
            return delta if delta > 0 else None
        except (TypeError, ValueError):
            return None

    # 1) History (config, timing, metrics, top-level)
    if isinstance(history, dict):
        for src in (
            history.get("config", {}) or {},
            history.get("timing", {}) or {},
            history.get("metrics", {}) or {},
            history,
        ):
            if isinstance(src, dict):
                found = _find(src) or _from_timestamps(src)
                if found is not None:
                    return found

        # 3) Sumar tiempos por época si existen
        for key in ("epoch_times", "epoch_durations", "epoch_times_s",
                    "times_per_epoch"):
            et = history.get(key) if isinstance(history, dict) else None
            if isinstance(et, (list, tuple)) and et:
                try:
                    total = float(sum(float(x) for x in et if x is not None))
                    if total > 0:
                        return total
                except (TypeError, ValueError):
                    pass

    # 2) run_info.json / metadata.json / run_metadata.json
    for meta_name in ("run_info.json", "metadata.json", "run_metadata.json"):
        meta_path = run_dir / meta_name
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        for src in (
            meta.get("config", {}) or {},
            meta.get("timing", {}) or {},
            meta.get("metrics", {}) or {},
            meta,
        ):
            if isinstance(src, dict):
                found = _find(src) or _from_timestamps(src)
                if found is not None:
                    return found

    return None

# ---------------------------------------------------------------------------
# Resolución de runs por ID (E08, 08, run_08, nombre completo)
# ---------------------------------------------------------------------------

def _resolve_run(root_runs: list[Path], token: str) -> Path | None:
    if not token:
        return None
    token = token.strip()

    for r in root_runs:
        if r.name == token:
            return r

    for r in root_runs:
        if r.name.startswith(token):
            return r

    t = token
    if t.lower().startswith("e") and t[1:].isdigit():
        num = t[1:].zfill(2)
        for r in root_runs:
            if f"run_{num}" in r.name:
                return r

    if t.isdigit():
        num = t.zfill(2)
        for r in root_runs:
            if f"run_{num}" in r.name:
                return r

    for r in root_runs:
        if token in r.name:
            return r

    return None


# ---------------------------------------------------------------------------
# Extracción de los 5 números clave
# ---------------------------------------------------------------------------

def extract_key_numbers(metrics: dict, diag: dict) -> dict:
    val = _safe(metrics.get("val_loss"))
    train = _safe(metrics.get("train_loss"))
    pos = _safe(metrics.get("val_pos_mean"))
    ori = _safe(metrics.get("val_ori_mean"))

    out = {
        "num_epochs": diag.get("num_epochs"),
        "best_val_epoch": None, "best_val_loss": None,
        "best_pos_epoch": None, "best_pos": None,
        "best_ori_epoch": None, "best_ori": None,
        "final_gap": None,
        "final_train_loss": None,
        "final_slope_val_loss": None,
    }

    if val is not None and not np.all(np.isnan(val)):
        idx = int(np.nanargmin(val))
        out["best_val_epoch"] = idx + 1
        out["best_val_loss"] = float(np.nanmin(val))
        n = min(20, len(val))
        if n >= 2:
            tail = val[-n:]
            if not np.all(np.isnan(tail)):
                x = np.arange(len(tail))
                out["final_slope_val_loss"] = float(np.polyfit(x, tail, 1)[0])

    if (val is not None and train is not None
            and len(val) > 0 and len(train) > 0):
        v, t = val[-1], train[-1]
        if not (np.isnan(v) or np.isnan(t)):
            out["final_gap"] = float(v - t)
            out["final_train_loss"] = float(t)

    if pos is not None and not np.all(np.isnan(pos)):
        idx = int(np.nanargmin(pos))
        out["best_pos_epoch"] = idx + 1
        out["best_pos"] = float(np.nanmin(pos))

    if ori is not None and not np.all(np.isnan(ori)):
        idx = int(np.nanargmin(ori))
        out["best_ori_epoch"] = idx + 1
        out["best_ori"] = float(np.nanmin(ori))

    return out


def summarize(run_dir: Path) -> tuple[dict, dict, dict]:
    history = load_history(run_dir)
    metrics = get_metrics(history)
    diag = detect_patterns(metrics)
    row = {"run": run_dir.name, "path": str(run_dir)}
    row.update(extract_key_numbers(metrics, diag))
    row["overfitting"]       = diag.get("overfitting")
    row["underfitting"]      = diag.get("underfitting")
    row["converged"]         = diag.get("converged")
    row["val_loss_rise_pct"] = diag.get("val_loss_rise_pct")
    row["training_time_s"]   = _extract_training_time(run_dir, history)
    return row, metrics, history


# ---------------------------------------------------------------------------
# Gráficas genéricas
# ---------------------------------------------------------------------------

def _horizontal_bar(summaries, key, title, xlabel, color, out_path, fmt="{:.4f}"):
    pairs = [(s["run"], s[key]) for s in summaries if s[key] is not None]
    if not pairs:
        return
    names, vals = zip(*sorted(pairs, key=lambda p: p[1]))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(names) + 3), 5))
    ax.barh(names, vals, color=color)
    for i, v in enumerate(vals):
        ax.text(v, i, "  " + fmt.format(v), va="center", fontsize=8)
    ax.set_title(title); ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_generic_val_loss(metrics_by_run, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, m in metrics_by_run.items():
        arr = _safe(m.get("val_loss"))
        if arr is None or np.all(np.isnan(arr)):
            continue
        ax.plot(_epochs_array(arr), arr, lw=1.5, label=name)
    ax.set_title("Validation loss por run")
    ax.set_xlabel("Época"); ax.set_ylabel("val_loss")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 1 — "El pipeline funciona"
# ---------------------------------------------------------------------------

def plot_fig_pipeline(root_runs, out_path):
    """E01 (baseline) -> E03 (primer competitivo) -> E08 (referencia)."""
    targets = [("E01", "baseline (bs2, 2 ép.)"),
               ("E03", "bs4, 80 ép."),
               ("E08", "bs8, LR 1e-4, 80 ép. (referencia)")]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = []
    for exp_id, label in targets:
        run = _resolve_run(root_runs, exp_id)
        if run is None:
            print(f"[!] Fig1: no se encontró {exp_id}; se omite.")
            continue
        metrics = get_metrics(load_history(run))
        val = _safe(metrics.get("val_loss"))
        if val is None or np.all(np.isnan(val)):
            continue
        ep = _epochs_array(val)
        ax.plot(ep, val, lw=1.8, label=f"{exp_id} — {label}")
        i = int(np.nanargmin(val))
        ax.scatter([ep[i]], [val[i]], s=40, zorder=5)
        ax.annotate(f"{val[i]:.4f}\n(ép. {ep[i]})",
                    (ep[i], val[i]), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)
        plotted.append(exp_id)

    if not plotted:
        plt.close(fig); return False
    ax.set_yscale("log")
    ax.set_title("Evolución del pipeline: baseline → primer competitivo → referencia")
    ax.set_xlabel("Época"); ax.set_ylabel("val_loss (escala log)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Figura 2 — Barrido de hiperparámetros
# ---------------------------------------------------------------------------

def plot_fig_sweep(root_runs, out_path):
    """Panel A: batch (E04 vs E05). Panel B: LR (E06, E08, E07)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for exp_id, label, color in [("E05", "E05 — batch 4", "tab:blue"),
                                  ("E04", "E04 — batch 8", "tab:orange")]:
        run = _resolve_run(root_runs, exp_id)
        if run is None:
            continue
        val = _safe(get_metrics(load_history(run)).get("val_loss"))
        if val is None or np.all(np.isnan(val)):
            continue
        ep = _epochs_array(val)
        ax.plot(ep, val, lw=1.8, color=color, label=label)
        i = int(np.nanargmin(val))
        ax.scatter([ep[i]], [val[i]], s=40, color=color, zorder=5)
        ax.annotate(f"{val[i]:.4f}", (ep[i], val[i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_title("A) Efecto del batch size (10 épocas)")
    ax.set_xlabel("Época"); ax.set_ylabel("val_loss")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for exp_id, label, color in [
        ("E06", "E06 — LR 5e-5", "tab:green"),
        ("E08", "E08 — LR 1e-4 (ref.)", "tab:red"),
        ("E07", "E07 — LR 2e-4", "tab:purple"),
    ]:
        run = _resolve_run(root_runs, exp_id)
        if run is None:
            continue
        val = _safe(get_metrics(load_history(run)).get("val_loss"))
        if val is None or np.all(np.isnan(val)):
            continue
        ep = _epochs_array(val)
        ax.plot(ep, val, lw=1.8, color=color, label=label)
        i = int(np.nanargmin(val))
        ax.scatter([ep[i]], [val[i]], s=40, color=color, zorder=5)
        ax.annotate(f"{val[i]:.4f}", (ep[i], val[i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_title("B) Efecto del learning rate")
    ax.set_xlabel("Época"); ax.set_ylabel("val_loss")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("Barrido sistemático: una variable a la vez", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 3 — Sobreajuste E08 vs E10 + línea límite
# ---------------------------------------------------------------------------

def plot_fig_overfitting(root_runs, out_path):
    """train y val de E08 (sólido) y E10 (punteado) + zona verde de no-overfit."""
    r08 = _resolve_run(root_runs, "E08")
    r10 = _resolve_run(root_runs, "E10")
    if r08 is None or r10 is None:
        print("[!] Fig3: faltan E08 y/o E10; se omite.")
        return False

    m08 = get_metrics(load_history(r08))
    m10 = get_metrics(load_history(r10))

    fig, ax = plt.subplots(figsize=(11, 6))

    def _plot(metrics, prefix, style, color):
        for key, label in (("train_loss", f"{prefix} train"),
                           ("val_loss", f"{prefix} val")):
            arr = _safe(metrics.get(key))
            if arr is None or np.all(np.isnan(arr)):
                continue
            ax.plot(_epochs_array(arr), arr, style, color=color, lw=1.6, label=label)

    _plot(m08, "E08", "-", "tab:blue")
    _plot(m10, "E10", "--", "tab:red")

    # Zona verde: val_loss ≤ train_loss + OVERFIT_GAP_THRESHOLD → no overfit
    t08 = _safe(m08.get("train_loss"))
    if t08 is not None and not np.all(np.isnan(t08)):
        ep08 = _epochs_array(t08)
        upper = t08 + OVERFIT_GAP_THRESHOLD
        ax.plot(ep08, upper, color="seagreen", lw=1.2, ls="--", alpha=0.85,
                label=f"límite overfit (E08 train + {OVERFIT_GAP_THRESHOLD})")
        ax.fill_between(ep08, t08, upper,
                        color="seagreen", alpha=0.08, label="_nolegend_")

    v08 = _safe(m08.get("val_loss"))
    n08 = _len(v08)
    if n08 > 0:
        ax.axvline(n08, color="black", lw=1, ls=":", alpha=0.7)
        ax.text(n08 + 2, ax.get_ylim()[1] * 0.9,
                f"fin de E08\n(ép. {n08})", fontsize=8, va="top")

    if v08 is not None and not np.all(np.isnan(v08)):
        i = int(np.nanargmin(v08))
        ax.scatter([i + 1], [v08[i]], s=60, color="black", zorder=6)
        ax.annotate(f"mejor val_loss\n{v08[i]:.6f} (ép. {i + 1})",
                    (i + 1, v08[i]), textcoords="offset points",
                    xytext=(10, 12), fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_title("Diagnóstico de sobreajuste: E08 vs E10\n"
                 "(mismas primeras 80 épocas por compartir seed 42)\n"
                 f"Zona verde = val_loss que NO se considera overfit "
                 f"(gap ≤ {OVERFIT_GAP_THRESHOLD})")
    ax.set_xlabel("Época"); ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Figura 4 — Mecanismos de control E13 / E14
# ---------------------------------------------------------------------------

def plot_fig_control(root_runs, out_path):
    """Panel A: E13 con LR superpuesto. Panel B: E13 vs E14."""
    r13 = _resolve_run(root_runs, "E13")
    r14 = _resolve_run(root_runs, "E14")
    if r13 is None:
        print("[!] Fig4: falta E13; se omite.")
        return False

    m13 = get_metrics(load_history(r13))
    h13 = load_history(r13)
    m14 = get_metrics(load_history(r14)) if r14 else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    v13 = _safe(m13.get("val_loss"))
    if v13 is not None and not np.all(np.isnan(v13)):
        ep13 = _epochs_array(v13)
        ax.plot(ep13, v13, lw=1.8, color="tab:blue", label="E13 val_loss")
        i = int(np.nanargmin(v13))
        ax.scatter([ep13[i]], [v13[i]], s=50, color="tab:blue", zorder=5)
        ax.annotate(f"mejor {v13[i]:.4f}\n(ép. {ep13[i]})",
                    (ep13[i], v13[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=8)

        lr, _ = _extract_lr(m13)
        if lr is not None:
            ax2 = ax.twinx()
            ax2.plot(_epochs_array(lr), lr, color="gray", lw=1, alpha=0.7,
                     label="LR")
            ax2.set_ylabel("learning rate", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")
            ax2.set_yscale("log")
            for ep in _detect_lr_drops(lr):
                ax.axvline(ep, color="gray", lw=0.8, ls="--", alpha=0.5)

        n_actual = len(ep13)
        cfg = _configured_epochs(h13)
        if cfg is not None and n_actual < cfg:
            ax.axvline(n_actual, color="red", lw=1.2, ls="-", alpha=0.7)
            ax.text(n_actual + 1, ax.get_ylim()[1] * 0.9,
                    f"early stop\n(ép. {n_actual}/{cfg})",
                    fontsize=8, color="red", va="top")
        else:
            ax.axvline(n_actual, color="red", lw=1, ls=":", alpha=0.6)

    ax.set_title("A) E13: scheduler + early stopping")
    ax.set_xlabel("Época"); ax.set_ylabel("val_loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, m, color in (("E13 (wd=0)", m13, "tab:blue"),
                             ("E14 (wd=1e-4)", m14, "tab:orange")):
        if m is None:
            continue
        v = _safe(m.get("val_loss"))
        if v is None or np.all(np.isnan(v)):
            continue
        ep = _epochs_array(v)
        ax.plot(ep, v, lw=1.8, color=color, label=label)
        i = int(np.nanargmin(v))
        ax.scatter([ep[i]], [v[i]], s=50, color=color, zorder=5)
        ax.annotate(f"{v[i]:.4f}\n(ép. {ep[i]})", (ep[i], v[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.set_title("B) E13 vs E14: efecto de weight decay")
    ax.set_xlabel("Época"); ax.set_ylabel("val_loss")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("Mecanismos de control del entrenamiento", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Figura 5 — Scatter trade-off val_loss vs error de pose
# ---------------------------------------------------------------------------

def plot_fig_tradeoff(summaries, out_path):
    """Un punto por run: (val_loss, mejor error de posición)."""
    pts = [(s["run"], s["best_val_loss"], s["best_pos"])
           for s in summaries
           if s["best_val_loss"] is not None and s["best_pos"] is not None]
    if not pts:
        print("[!] Fig5: no hay runs con val_loss y posición; se omite.")
        return False

    names, vls, poss = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(vls, poss, s=60, color="tab:purple", zorder=5)
    for name, x, y in zip(names, vls, poss):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)
    ax.set_xlabel("mejor val_loss"); ax.set_ylabel("mejor error de posición (m)")
    ax.set_title("Trade-off: val_loss vs error de posición\n"
                 "(cada punto es un run, no un checkpoint)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Figura 5b — Diagnóstico overfit/underfit por run (rejilla)
# ---------------------------------------------------------------------------

def _diagnostic_reason(s: dict, tag: str) -> str:
    """Justificación numérica del tag asignado a un run (para el subtítulo)."""
    if tag == "OVERFIT":
        parts = []
        gap = s.get("final_gap")
        if gap is not None:
            parts.append(f"gap={gap:+.3f}")
        rise = s.get("val_loss_rise_pct")
        if rise is not None:
            parts.append(f"sube {rise:.1f}%")
        return ", ".join(parts)
    if tag == "UNDERFIT":
        train_f = s.get("final_train_loss")
        if train_f is not None:
            return f"train final={train_f:.3f}"
        return ""
    if tag == "NO CONV.":
        slope = s.get("final_slope_val_loss")
        if slope is not None:
            return f"slope={slope:+.4f}/ép"
        return ""
    return ""


def plot_overfit_underfit_diagnostic(metrics_by_run, summaries, out_path):
    """Rejilla small-multiples: train vs val + gap + referencias + tag.

    Referencias dibujadas en cada panel:
      - verde discontinua:  train + OVERFIT_GAP_THRESHOLD (val por debajo → OK)
      - gris punteada:      UNDERFIT_MIN_LOSS (train+val altos → underfit)
    """
    runs = []
    for name, m in metrics_by_run.items():
        train = _safe(m.get("train_loss"))
        val   = _safe(m.get("val_loss"))
        if train is None or val is None:
            continue
        if np.all(np.isnan(train)) and np.all(np.isnan(val)):
            continue
        runs.append((name, train, val))
    if not runs:
        print("[!] Diag: sin curvas train/val válidas; se omite.")
        return False

    n     = len(runs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        squeeze=False,
    )

    diag_by_name = {s["run"]: s for s in summaries}

    for idx, (name, train, val) in enumerate(runs):
        ax   = axes[idx // ncols][idx % ncols]
        n_ep = min(_len(train), _len(val))
        if n_ep == 0:
            _empty_panel(ax, name); continue
        t, v = train[:n_ep], val[:n_ep]
        ep   = np.arange(1, n_ep + 1)
        gap  = v - t

        ax.plot(ep, t, color="tab:blue", lw=1.6, label="train")
        ax.plot(ep, v, color="tab:red",  lw=1.6, label="val")
        ax.fill_between(ep, t, v, where=~np.isnan(gap),
                        color="purple", alpha=0.15, label="gap")

        # Límite overfit: verde = hasta aquí es seguro
        safe_upper = t + OVERFIT_GAP_THRESHOLD
        ax.plot(ep, safe_upper, color="seagreen", lw=1.0, ls="--", alpha=0.85,
                label=f"límite overfit (+{OVERFIT_GAP_THRESHOLD})")
        ax.fill_between(ep, t, safe_upper,
                        color="seagreen", alpha=0.07, label="_nolegend_")

        # Umbral underfit: si train+val terminan por encima, sospecha
        ax.axhline(UNDERFIT_MIN_LOSS, color="gray", lw=0.8, ls=":",
                   alpha=0.6, label=f"umbral underfit ({UNDERFIT_MIN_LOSS})")

        ax.axhline(0, color="black", lw=0.6, ls=":")

        if not np.all(np.isnan(v)):
            i = int(np.nanargmin(v))
            ax.scatter([ep[i]], [v[i]], s=50, color="tab:red",
                       edgecolor="black", zorder=5, linewidth=0.7)

        s = diag_by_name.get(name, {})
        if s.get("overfitting"):
            tag, color = "OVERFIT", "crimson"
        elif s.get("underfitting"):
            tag, color = "UNDERFIT", "darkorange"
        elif not s.get("converged", True):
            tag, color = "NO CONV.", "orange"
        else:
            tag, color = "OK", "seagreen"

        reason = _diagnostic_reason(s, tag)
        title  = f"{name} — {tag}"
        if reason:
            title += f"\n{reason}"

        ax.set_title(title, fontsize=9, color=color, fontweight="bold")
        ax.set_xlabel("Época", fontsize=8)
        ax.set_ylabel("loss", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

    for idx in range(len(runs), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        "Diagnóstico overfitting / underfitting por run\n"
        f"verde = límite overfit (gap ≤ {OVERFIT_GAP_THRESHOLD}); "
        f"gris = umbral underfit (train_loss ≥ {UNDERFIT_MIN_LOSS})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Figura 6 — Comparación directa de 2 runs
# ---------------------------------------------------------------------------

def plot_pair_compare(root_runs, id_a, id_b, out_path):
    """Overlay train/val + gap con umbral overfit para 2 runs."""
    ra = _resolve_run(root_runs, id_a)
    rb = _resolve_run(root_runs, id_b)
    if ra is None or rb is None:
        missing = [x for x, r in ((id_a, ra), (id_b, rb)) if r is None]
        print(f"[!] Pair: no se encontraron {missing}; se omite.")
        return False

    ma = get_metrics(load_history(ra))
    mb = get_metrics(load_history(rb))

    fig, (ax_loss, ax_gap) = plt.subplots(
        1, 2, figsize=(15, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    palette = {ra.name: "tab:blue", rb.name: "tab:red"}
    style   = {ra.name: "-",        rb.name: "--"}

    best_info = []
    for name, m in ((ra.name, ma), (rb.name, mb)):
        train = _safe(m.get("train_loss"))
        val   = _safe(m.get("val_loss"))
        if train is None or val is None:
            continue
        color, ls = palette[name], style[name]
        ax_loss.plot(_epochs_array(train), train, color=color, ls=ls,
                     lw=1.8, label=f"{name} train")
        ax_loss.plot(_epochs_array(val),   val,   color=color, ls=":",
                     lw=1.8, label=f"{name} val")

        n = min(len(train), len(val))
        if n > 0:
            gap = val[:n] - train[:n]
            ax_gap.plot(np.arange(1, n + 1), gap, color=color, ls=ls,
                        lw=1.6, label=f"{name} gap")

        if not np.all(np.isnan(val)):
            i = int(np.nanargmin(val))
            ax_loss.scatter([i + 1], [val[i]], s=70, color=color,
                            edgecolor="black", linewidth=0.8, zorder=5)
            best_info.append(f"{name}: mejor val={val[i]:.4f} @ép.{i+1}")

    ax_loss.set_title("Train vs Val loss")
    ax_loss.set_xlabel("Época"); ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3); ax_loss.legend(fontsize=8)

    # Umbral overfit en el panel del gap
    ax_gap.axhline(0, color="black", lw=0.7, ls=":")
    ax_gap.axhline(OVERFIT_GAP_THRESHOLD, color="seagreen", lw=1.2, ls="--",
                   alpha=0.9,
                   label=f"umbral overfit ({OVERFIT_GAP_THRESHOLD})")
    ax_gap.axhspan(0, OVERFIT_GAP_THRESHOLD, color="seagreen", alpha=0.08,
                   label="_nolegend_")
    ax_gap.set_title("Gap = val − train")
    ax_gap.set_xlabel("Época"); ax_gap.set_ylabel("gap")
    ax_gap.grid(True, alpha=0.3); ax_gap.legend(fontsize=8)

    fig.suptitle(
        f"Comparación directa: {ra.name} vs {rb.name}\n"
        + " | ".join(best_info)
        + f"\n(gap > {OVERFIT_GAP_THRESHOLD} ⇒ overfit; zona verde = seguro)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Figura 7 — Comparación de tiempos de ejecución
# ---------------------------------------------------------------------------

def plot_training_times(summaries, out_path):
    """Panel A: tiempo total de entrenamiento por run.
    Panel B: tiempo por época (normaliza runs con distinto nº de épocas)."""
    pairs = []
    for s in summaries:
        t = s.get("training_time_s")
        if t is None:
            continue
        ne = s.get("num_epochs")
        per_epoch = (t / ne) if (isinstance(ne, (int, float)) and ne > 0) else None
        pairs.append((s["run"], float(t), per_epoch))

    if not pairs:
        print("[!] Tiempos: ningún run tiene training_time registrado; se omite.")
        return False

    pairs_total  = sorted([(n, t) for n, t, _ in pairs], key=lambda p: p[1])
    pairs_per_ep = sorted([(n, e) for n, _, e in pairs if e is not None],
                          key=lambda p: p[1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    def _draw(ax, data, xlabel, title, unit_fmt, line_label_fmt):
        names, vals = zip(*data)
        v_min, v_max = float(min(vals)), float(max(vals))
        if v_max > v_min:
            colors = [plt.cm.RdYlGn_r((v - v_min) / (v_max - v_min)) for v in vals]
        else:
            colors = ["steelblue"] * len(vals)
        ax.barh(names, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(v, i, "  " + unit_fmt(v), va="center", fontsize=8)
        mean = float(np.mean(vals))
        ax.axvline(mean, color="black", lw=1, ls="--", alpha=0.7,
                   label=line_label_fmt(mean))
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    # Panel A: total
    _draw(axes[0], pairs_total,
          xlabel="Tiempo total (s)",
          title="A) Tiempo total de entrenamiento",
          unit_fmt=_fmt_time,
          line_label_fmt=lambda m: f"media = {_fmt_time(m)}")

    # Panel B: por época
    if pairs_per_ep:
        _draw(axes[1], pairs_per_ep,
              xlabel="Segundos por época",
              title="B) Tiempo por época (normaliza por nº de épocas)",
              unit_fmt=lambda v: f"{v:.2f}s",
              line_label_fmt=lambda m: f"media = {m:.2f}s")
    else:
        _empty_panel(axes[1], "B) Tiempo por época")

    fig.suptitle("Comparación de tiempos de ejecución por entrenamiento",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check_e08_vs_e10(root_runs) -> str:
    r08 = _resolve_run(root_runs, "E08")
    r10 = _resolve_run(root_runs, "E10")
    if r08 is None or r10 is None:
        return "E08 vs E10: **faltan runs**, no se pudo verificar.\n"

    v08 = _safe(get_metrics(load_history(r08)).get("val_loss"))
    v10 = _safe(get_metrics(load_history(r10)).get("val_loss"))
    if v08 is None or v10 is None:
        return "E08 vs E10: sin curva de val_loss, no se pudo verificar.\n"

    n = min(len(v08), len(v10), 80)
    a, b = v08[:n], v10[:n]
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() == 0:
        return "E08 vs E10: tramo 1–80 con NaN, no se pudo verificar.\n"

    max_abs = float(np.nanmax(np.abs(a[mask] - b[mask])))
    ok = max_abs < 1e-6
    lines = [
        "## Sanity check E08 vs E10\n",
        f"- Tramo comparado: épocas 1–{n}.",
        f"- Máxima diferencia absoluta en val_loss: **{max_abs:.3e}**.",
        f"- Veredicto: **{'OK — idénticos (misma seed, misma config)' if ok else 'DIFERENTES — revisar reproducibilidad'}**.",
    ]
    if len(v10) > 80:
        div = v10[80:]
        if not np.all(np.isnan(div)):
            lines.append(
                f"- A partir de la época 81, E10 sigue: `val_loss` final "
                f"= {np.nan_to_num(div[-1]):.4f} (debería ser > mínimo)."
            )
    return "\n".join(lines) + "\n"


def check_e13_e14(root_runs) -> str:
    lines = ["## Sanity check E13 / E14 (scheduler + early stop)\n"]
    for exp_id in ("E13", "E14"):
        run = _resolve_run(root_runs, exp_id)
        if run is None:
            lines.append(f"- **{exp_id}**: no encontrado.\n")
            continue
        history = load_history(run)
        metrics = get_metrics(history)
        n_actual = _len(_safe(metrics.get("val_loss")))
        cfg = _configured_epochs(history)
        lr, lr_key = _extract_lr(metrics)
        drops = _detect_lr_drops(lr) if lr is not None else []
        lines.append(f"### {exp_id}\n")
        lines.append(f"- Épocas reales: **{n_actual}**"
                     + (f" (configuradas: {cfg})" if cfg else ""))
        if cfg and n_actual < cfg:
            lines.append(f"- Early stopping detectado: "
                         f"**ép. {n_actual}/{cfg}**.")
        elif cfg:
            lines.append("- Early stopping **no** detectado "
                         "(se agotaron las épocas configuradas).")
        else:
            lines.append("- No se pudo leer el número de épocas configuradas "
                         "del history; verificar manualmente.")
        if lr is not None:
            lines.append(f"- Curva de LR encontrada en key `{lr_key}`.")
            lines.append(f"- Caídas de LR detectadas en épocas: "
                         f"**{drops if drops else 'ninguna'}**.")
            lines.append(f"- LR inicial: {lr[0]:.2e} → final: {lr[-1]:.2e}.")
        else:
            lines.append("- ⚠️ No se encontró curva de LR en el history "
                         "(buscar en `config` o en logs).")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Informe numérico
# ---------------------------------------------------------------------------

def write_numeric_summary(summaries, out_path: Path):
    lines = ["# Extracción numérica por run\n",
             "Los 5 números clave de cada curva (+ tiempo total):\n",
             "| Run | Épocas | Tiempo | Mejor val_loss | Ép. | Mejor pos (m) | Ép. | "
             "Mejor ori (°) | Ép. | Gap final | Pendiente val (últ. 20) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(
            f"| {s['run']} | {_fmt(s['num_epochs'], 'd')} | "
            f"{_fmt_time(s.get('training_time_s'))} | "
            f"{_fmt(s['best_val_loss'], '.6f')} | {_fmt(s['best_val_epoch'], 'd')} | "
            f"{_fmt(s['best_pos'])} | {_fmt(s['best_pos_epoch'], 'd')} | "
            f"{_fmt(s['best_ori'])} | {_fmt(s['best_ori_epoch'], 'd')} | "
            f"{_fmt(s['final_gap'])} | {_fmt(s['final_slope_val_loss'], '+.5f')} |"
        )
    lines.append("\n**Umbrales usados para clasificar (fuente: `src/analysis_nn/__init__.py`):**\n")
    lines.append(f"- `OVERFIT_GAP_THRESHOLD = {OVERFIT_GAP_THRESHOLD}`")
    lines.append(f"- `OVERFIT_RISE_PCT = {OVERFIT_RISE_PCT}`")
    lines.append(f"- `UNDERFIT_MIN_LOSS = {UNDERFIT_MIN_LOSS}`")
    lines.append(f"- `UNDERFIT_DROP_PCT = {UNDERFIT_DROP_PCT}`")
    lines.append(f"- `CONVERGED_SLOPE_EPS = {CONVERGED_SLOPE_EPS}`")
    lines.append(f"- `CONVERGED_CV_EPS = {CONVERGED_CV_EPS}`")
    lines.append("\n**Lectura:**\n")
    lines.append("- `Gap final` = `val_loss` final − `train_loss` final. "
                 "Positivo grande → sobreajuste.")
    lines.append("- `Pendiente val` > 0 en las últimas 20 épocas → "
                 "el modelo ya no generaliza; < 0 → aún aprendía.")
    lines.append("- `Ép.` junto a cada métrica indica si el mejor valor "
                 "coincide con el mejor `val_loss` o no.")
    lines.append("- `Tiempo` = tiempo total de entrenamiento extraído del "
                 "history o de `metadata.json` (vacío si no está registrado).")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Selección de runs (CLI)
# ---------------------------------------------------------------------------

def _select_runs(parser, args, root_runs) -> list[Path]:
    def resolve_or_die(token: str) -> Path:
        r = _resolve_run(root_runs, token)
        if r is None:
            parser.error(f"Run '{token}' no encontrado bajo {args.root}")
        return r

    if args.run:
        selected = [resolve_or_die(t) for t in args.run]
    elif args.last:
        if args.last < 1:
            parser.error("--last debe ser >= 1")
        selected = find_latest_runs(args.root, args.last)
    else:
        selected = list(root_runs)
        if not args.exclude:
            print(f"[i] Sin filtros: comparando TODOS los runs ({len(selected)}). "
                  f"Usa --run / --last / --exclude para elegir.")

    if args.exclude:
        excluded = {resolve_or_die(t).resolve() for t in args.exclude}
        selected = [r for r in selected if r.resolve() not in excluded]

    seen, unique = set(), []
    for r in selected:
        k = r.resolve()
        if k not in seen:
            seen.add(k); unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara runs de entrenamiento y genera figuras para asesoría.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python -m src.neuralNetwork.analysis.compare_runs outputs_nn/resnet18\n"
            "  python -m src.neuralNetwork.analysis.compare_runs outputs_nn/resnet18 --run E08 --run E10\n"
            "  python -m src.neuralNetwork.analysis.compare_runs outputs_nn/resnet18 --last 3\n"
            "  python -m src.neuralNetwork.analysis.compare_runs outputs_nn/resnet18 --exclude E02\n"
            "  python -m src.neuralNetwork.analysis.compare_runs outputs_nn/resnet18 --pair E06 E07\n"
        ),
    )
    parser.add_argument("root", help="Directorio raíz con varios runs.")
    parser.add_argument("--run", action="append", default=[], metavar="RUN",
                        help="Run a incluir (E08, 08, run_08, nombre completo; repetible).")
    parser.add_argument("--last", type=int, default=None, metavar="N",
                        help="Incluir solo los N runs más recientes.")
    parser.add_argument("--exclude", action="append", default=[], metavar="RUN",
                        help="Run a excluir (mismo formato que --run).")
    parser.add_argument("--out", default=None, help="Directorio de salida.")
    parser.add_argument("--no-curated", action="store_true",
                        help="No generar las figuras curadas (solo comparación genérica).")
    parser.add_argument("--pair", nargs=2, default=None, metavar=("A", "B"),
                        help=f"Comparar solo estos 2 runs en una figura extra "
                             f"(default: {PAIR_DEFAULT[0]} vs {PAIR_DEFAULT[1]}).")
    parser.add_argument("--no-diag-grid", action="store_true",
                        help="No generar la rejilla overfit/underfit por run.")
    args = parser.parse_args()

    root = Path(args.root)
    root_runs = discover_runs(root)
    if not root_runs:
        parser.error(f"No se encontraron history.json bajo {root}")

    runs = _select_runs(parser, args, root_runs)
    if not runs:
        parser.error("La selección quedó vacía tras aplicar los filtros.")
    if len(runs) == 1:
        print(f"[!] Solo hay 1 run tras filtrar ({runs[0].name}); "
              f"las comparativas serán triviales.")

    # --exclude afecta a las figuras curadas EXCEPTO Fig1 (que necesita
    # E01/E03/E08 aunque estén excluidos del reporte).
    excluded_paths = set()
    for t in (args.exclude or []):
        r = _resolve_run(root_runs, t)
        if r is not None:
            excluded_paths.add(r.resolve())
    root_runs_for_figs = [r for r in root_runs if r.resolve() not in excluded_paths]

    out_dir = Path(args.out) if args.out else root / "analysis_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, metrics_by_run = [], {}
    for run in runs:
        try:
            row, m, _h = summarize(run)
            summaries.append(row)
            metrics_by_run[row["run"]] = m
        except Exception as exc:
            print(f"[!] Saltando {run}: {exc}")

    if not summaries:
        print("[!] No hay runs válidos.")
        return

    # --- Figuras genéricas ---
    plot_generic_val_loss(metrics_by_run, out_dir / "01_val_loss_compare.png")
    _horizontal_bar(summaries, "best_val_loss",
                    "Mejor val_loss por run (menor = mejor)",
                    "val_loss", "steelblue",
                    out_dir / "02_best_val_loss.png")
    _horizontal_bar(summaries, "best_pos",
                    "Mejor error medio de posición (val)",
                    "metros", "seagreen",
                    out_dir / "03_best_position_error.png")
    _horizontal_bar(summaries, "best_ori",
                    "Mejor error medio de orientación (val)",
                    "grados", "darkorange",
                    out_dir / "04_best_orientation_error.png")

    # --- Extracción numérica + sanity checks ---
    write_numeric_summary(summaries, out_dir / "05_numeric_summary.md")
    sanity = [check_e08_vs_e10(root_runs), check_e13_e14(root_runs)]
    (out_dir / "06_sanity_checks.md").write_text(
        "\n".join(sanity), encoding="utf-8")

    # --- Figuras curadas ---
    if not args.no_curated:
        # Fig1 siempre usa root_runs (necesita E01/E03/E08 aunque excluyas)
        plot_fig_pipeline(root_runs,             out_dir / "07_fig1_pipeline_overview.png")
        plot_fig_sweep(root_runs_for_figs,       out_dir / "08_fig2_hyperparam_sweep.png")
        plot_fig_overfitting(root_runs_for_figs, out_dir / "09_fig3_overfitting.png")
        plot_fig_control(root_runs_for_figs,     out_dir / "10_fig4_control_mechanisms.png")
        plot_fig_tradeoff(summaries,             out_dir / "11_fig5_tradeoff_scatter.png")

    # --- Rejilla overfit/underfit ---
    if not args.no_diag_grid:
        plot_overfit_underfit_diagnostic(
            metrics_by_run, summaries,
            out_dir / "12_fig5b_overfit_diagnostic.png",
        )

    # --- Par de runs ---
    pair = tuple(args.pair) if args.pair else PAIR_DEFAULT
    plot_pair_compare(
        root_runs, pair[0], pair[1],
        out_dir / "13_fig6_pair_compare.png",
    )

    # --- Tiempos de ejecución ---
    plot_training_times(
        summaries,
        out_dir / "14_fig7_training_times.png",
    )

    # --- report.md consolidado ---
    lines = [
        "# Comparación de runs\n",
        f"Runs comparados: **{len(summaries)}**\n",
        "## Umbrales de diagnóstico\n",
        "| Constante | Valor | Uso |",
        "|---|---:|---|",
        f"| `OVERFIT_GAP_THRESHOLD` | {OVERFIT_GAP_THRESHOLD} | gap (val−train) > esto → overfit |",
        f"| `OVERFIT_RISE_PCT` | {OVERFIT_RISE_PCT}% | subida de val tras el mínimo para confirmar overfit |",
        f"| `UNDERFIT_MIN_LOSS` | {UNDERFIT_MIN_LOSS} | train_loss final > esto → posible underfit |",
        f"| `UNDERFIT_DROP_PCT` | {UNDERFIT_DROP_PCT}% | bajada mínima de train para no ser underfit |",
        f"| `CONVERGED_SLOPE_EPS` | {CONVERGED_SLOPE_EPS} | |slope|(cola) < esto → convergido |",
        f"| `CONVERGED_CV_EPS` | {CONVERGED_CV_EPS} | coef. variación (sd/mu) de la cola < esto → convergido |",
        "",
        "> Fuente única de verdad: `src/analysis_nn/__init__.py`. "
        "Cambiar allí y todo el pipeline queda consistente.",
        "",
        "## Tabla resumen\n",
        "| Run | Épocas | Tiempo | Mejor val_loss | Ép. | Mejor pos (m) | "
        "Mejor ori (°) | Gap final | Overfit | Underfit | Conv. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['run']} | {_fmt(s['num_epochs'], 'd')} | "
            f"{_fmt_time(s.get('training_time_s'))} | "
            f"{_fmt(s['best_val_loss'], '.6f')} | {_fmt(s['best_val_epoch'], 'd')} | "
            f"{_fmt(s['best_pos'])} | {_fmt(s['best_ori'])} | "
            f"{_fmt(s['final_gap'])} | "
            f"{'sí' if s['overfitting'] else 'no'} | "
            f"{'sí' if s['underfitting'] else 'no'} | "
            f"{'sí' if s['converged'] else 'no'} |"
        )

    for label, key, unit in (
        ("Ganador por val_loss",             "best_val_loss", ""),
        ("Ganador por error de posición",    "best_pos",      " m"),
        ("Ganador por error de orientación", "best_ori",      "°"),
    ):
        valid = [s for s in summaries if s[key] is not None]
        if valid:
            best = min(valid, key=lambda s: s[key])
            lines.append(f"\n## {label}\n- `{best['run']}` con **{best[key]:.4f}{unit}**")

    # --- Sección de tiempos ---
    with_t = [s for s in summaries if s.get("training_time_s") is not None]
    if with_t:
        fastest = min(with_t, key=lambda s: s["training_time_s"])
        slowest = max(with_t, key=lambda s: s["training_time_s"])
        mean_t  = float(np.mean([s["training_time_s"] for s in with_t]))
        lines.append(
            "\n## Tiempo de entrenamiento\n"
            f"- Runs con tiempo registrado: **{len(with_t)}/{len(summaries)}**.\n"
            f"- Más rápido: `{fastest['run']}` con "
            f"**{_fmt_time(fastest['training_time_s'])}**.\n"
            f"- Más lento:  `{slowest['run']}` con "
            f"**{_fmt_time(slowest['training_time_s'])}**.\n"
            f"- Media: **{_fmt_time(mean_t)}**."
        )
    else:
        lines.append(
            "\n## Tiempo de entrenamiento\n"
            "- ⚠️ Ningún run tiene `training_time_seconds` (ni similar) "
            "en su history/metadata. Añade esa clave para que aparezca."
        )

    lines.append("\n## Figuras generadas\n")
    for fname, desc in [
        ("01_val_loss_compare.png", "Curvas val_loss (genérica, escala log + zoom)"),
        ("02_best_val_loss.png",    "Ranking por mejor val_loss"),
        ("03_best_position_error.png", "Ranking por mejor error de posición"),
        ("04_best_orientation_error.png", "Ranking por mejor error de orientación"),
        ("05_numeric_summary.md",   "Los 5 números clave por run (+ tiempo)"),
        ("06_sanity_checks.md",     "Checks E08/E10 y E13/E14"),
        ("07_fig1_pipeline_overview.png", "FIG1 — Pipeline: E01 → E03 → E08"),
        ("08_fig2_hyperparam_sweep.png",  "FIG2 — Batch (A) y LR (B)"),
        ("09_fig3_overfitting.png", "FIG3 — Sobreajuste E08 vs E10 + línea límite"),
        ("10_fig4_control_mechanisms.png", "FIG4 — E13 (scheduler+early) y E14 (+wd)"),
        ("11_fig5_tradeoff_scatter.png",  "FIG5 — Trade-off val_loss vs posición"),
        ("12_fig5b_overfit_diagnostic.png", "FIG5b — Diagnóstico por run con líneas límite"),
        ("13_fig6_pair_compare.png",        "FIG6 — Comparación de 2 runs con umbral overfit"),
        ("14_fig7_training_times.png",      "FIG7 — Tiempos de ejecución (total y por época)"),
    ]:
        if (out_dir / fname).exists():
            lines.append(f"- `{fname}` — {desc}")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[✓] Comparación generada en {out_dir}")
    print(f"    - Lee primero: {out_dir / '06_sanity_checks.md'}")
    print(f"    - Figura clave: {out_dir / '09_fig3_overfitting.png'}")
    print(f"    - Diagnóstico general: {out_dir / '12_fig5b_overfit_diagnostic.png'}")
    print(f"    - Par {pair[0]} vs {pair[1]}: {out_dir / '13_fig6_pair_compare.png'}")
    print(f"    - Tiempos: {out_dir / '14_fig7_training_times.png'}")


if __name__ == "__main__":
    main()