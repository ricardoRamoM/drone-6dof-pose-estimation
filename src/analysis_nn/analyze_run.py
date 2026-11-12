"""Analiza uno o varios entrenamientos: 5 gráficas + resumen 2x3 + config + informe.

Salida por run (por defecto):
    <root>/analysis/runs/<run_name>/
        ├── 01_loss.png
        ├── 02_position_error.png
        ├── 03_orientation_error.png
        ├── 04_learning_rate.png        (fallback a línea constante si no hay curva)
        ├── 05_generalization_gap.png
        ├── 06_summary.png              (dashboard + franja de configuración)
        └── report.md

Copia centralizada de todos los dashboards:
    <root>/analysis/summaries/<run_name>.png

Reporte consolidado (nuevo):
    <analysis>/all_reports.md           concatena todos los report.md con índice
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis_nn import (
    load_history, get_metrics, discover_runs, detect_patterns,
    find_latest_run, get_config,
    # Umbrales compartidos (fuente única: src/analysis_nn/__init__.py)
    OVERFIT_GAP_THRESHOLD, OVERFIT_RISE_PCT,
    UNDERFIT_MIN_LOSS, UNDERFIT_DROP_PCT,
    CONVERGED_SLOPE_EPS, CONVERGED_CV_EPS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(values):
    if values is None:
        return None
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def _plot(ax, values, label, **kwargs) -> bool:
    arr = _safe(values)
    if arr is None or arr.size == 0 or np.all(np.isnan(arr)):
        return False
    ax.plot(np.arange(1, len(arr) + 1), arr, label=label, **kwargs)
    return True


def _empty(ax, title, msg="Sin datos"):
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="gray")


def _fmt(v, spec=None) -> str:
    if v is None or v == "":
        return "—"
    try:
        return format(v, spec) if spec else str(v)
    except (ValueError, TypeError):
        return str(v)


def _duration_str(sec) -> str:
    if sec is None:
        return "—"
    try:
        return str(timedelta(seconds=round(float(sec), 1)))
    except (ValueError, TypeError):
        return "—"


# ---------------------------------------------------------------------------
# Dibujo: individuales
# ---------------------------------------------------------------------------

def _draw_loss(ax, metrics, diag) -> bool:
    plotted  = _plot(ax, metrics["train_loss"], "train_loss", marker="o", ms=3, lw=1.5)
    plotted |= _plot(ax, metrics["val_loss"],   "val_loss",   marker="s", ms=3, lw=1.5)

    # Límite overfit: train + OVERFIT_GAP_THRESHOLD
    train_arr = _safe(metrics["train_loss"])
    if train_arr is not None and train_arr.size and not np.all(np.isnan(train_arr)):
        ep = np.arange(1, len(train_arr) + 1)
        upper = train_arr + OVERFIT_GAP_THRESHOLD
        ax.plot(ep, upper, color="seagreen", lw=1.0, ls="--", alpha=0.85,
                label=f"límite overfit (train + {OVERFIT_GAP_THRESHOLD})")
        ax.fill_between(ep, train_arr, upper,
                        color="seagreen", alpha=0.07, label="_nolegend_")

    if "best_val_loss_epoch" in diag:
        ax.axvline(diag["best_val_loss_epoch"], color="gray", ls="--", lw=1,
                   label=f"mejor val (ep {diag['best_val_loss_epoch']})")
    if not plotted:
        _empty(ax, "Train Loss vs Validation Loss"); return False
    ax.set_title("Train Loss vs Validation Loss")
    ax.set_xlabel("Época"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    return True


def _draw_gap(ax, metrics) -> bool:
    train, val = _safe(metrics["train_loss"]), _safe(metrics["val_loss"])
    if train is None or val is None or train.size == 0 or val.size == 0:
        _empty(ax, "Generalization gap"); return False
    n = min(len(train), len(val))
    gap = val[:n] - train[:n]
    ep = np.arange(1, n + 1)
    ax.plot(ep, gap, color="purple", lw=1.5, label="val_loss − train_loss")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    # Zona segura: gap ≤ OVERFIT_GAP_THRESHOLD
    ax.axhline(OVERFIT_GAP_THRESHOLD, color="seagreen", lw=1.2, ls="--",
               alpha=0.85,
               label=f"límite overfit ({OVERFIT_GAP_THRESHOLD})")
    ax.axhspan(0, OVERFIT_GAP_THRESHOLD, color="seagreen", alpha=0.08,
               label="_nolegend_")
    ax.fill_between(ep, 0, gap, alpha=0.15, color="purple")
    ax.set_title("Generalization gap")
    ax.set_xlabel("Época"); ax.set_ylabel("gap")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    return True


def _draw_position(ax, metrics) -> bool:
    plotted  = _plot(ax, metrics["train_pos_mean"],   "train · media")
    plotted |= _plot(ax, metrics["val_pos_mean"],     "val · media")
    plotted |= _plot(ax, metrics["train_pos_median"], "train · mediana", ls="--")
    plotted |= _plot(ax, metrics["val_pos_median"],   "val · mediana",   ls="--")
    if not plotted:
        _empty(ax, "Error de posición"); return False
    ax.set_title("Error de posición")
    ax.set_xlabel("Época"); ax.set_ylabel("metros")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    return True


def _draw_orientation(ax, metrics) -> bool:
    plotted  = _plot(ax, metrics["train_ori_mean"],   "train · media")
    plotted |= _plot(ax, metrics["val_ori_mean"],     "val · media")
    plotted |= _plot(ax, metrics["train_ori_median"], "train · mediana", ls="--")
    plotted |= _plot(ax, metrics["val_ori_median"],   "val · mediana",   ls="--")
    if not plotted:
        _empty(ax, "Error de orientación"); return False
    ax.set_title("Error de orientación")
    ax.set_xlabel("Época"); ax.set_ylabel("grados")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    return True


def _draw_lr(ax, metrics, config) -> bool:
    # 1) serie real de learning_rate en history.json
    if _plot(ax, metrics.get("lr"), "learning_rate"):
        ax.set_yscale("log")
        ax.set_title("Learning Rate")
        ax.set_xlabel("Época"); ax.set_ylabel("lr (log)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        return True

    # 2) fallback: LR del run_info.json o del nombre de carpeta
    initial_lr = config.get("lr")
    final_lr   = config.get("final_learning_rate")
    n          = int(config.get("epochs") or 1)
    if initial_lr is None:
        _empty(ax, "Learning Rate",
               "Sin learning_rate\nen history.json\nni en run_info.json")
        return False

    ax.axhline(initial_lr, color="C0", lw=1.6,
               label=f"lr inicial (config) = {initial_lr:.3g}")
    if final_lr is not None and final_lr != initial_lr:
        ax.axhline(final_lr, color="C1", lw=1.2, ls="--",
                   label=f"lr final (config) = {final_lr:.3g}")
    ax.set_yscale("log")
    ax.set_xlim(1, max(n, 2))
    ax.set_title("Learning Rate (aprox. — sin curva por época)")
    ax.set_xlabel("Época"); ax.set_ylabel("lr (log)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    return True


# ---------------------------------------------------------------------------
# Dibujo: textos del dashboard
# ---------------------------------------------------------------------------

def _draw_status_text(ax, diag, metrics) -> None:
    ax.axis("off")
    ax.set_title("Métricas clave (validación)", fontsize=11)
    if "error" in diag:
        ax.text(0.5, 0.5, f"⚠️  {diag['error']}",
                ha="center", va="center", transform=ax.transAxes, wrap=True)
        return
    lines = [
        f"Épocas         : {diag['num_epochs']}",
        "",
        f"Mejor val_loss : {diag['best_val_loss']:.6f}",
        f"   (época {diag['best_val_loss_epoch']})",
        f"val_loss final : {diag['final_val_loss']:.6f}",
        f"train_loss final: {diag['final_train_loss']:.6f}",
        f"gap final      : {diag['generalization_gap_final']:.6f}",
        f"gap máximo     : {diag['generalization_gap_max']:.6f}",
    ]
    if diag.get("val_loss_rise_pct") is not None:
        lines.append(f"subida val tras mejor: {diag['val_loss_rise_pct']:.2f}%")
    if diag.get("final_slope_val_loss") is not None:
        lines.append(f"pendiente val (cola): {diag['final_slope_val_loss']:+.5f}/ép")
    for key, label, unit in (
        ("val_pos_mean", "Mejor error posición   ", "m"),
        ("val_ori_mean", "Mejor error orientación", "°"),
    ):
        arr = _safe(metrics.get(key))
        if arr is not None and arr.size and not np.all(np.isnan(arr)):
            i = int(np.nanargmin(arr))
            lines.append(f"{label}: {arr[i]:.4f} {unit} (ep {i+1})")
    lines += [
        "",
        f"Overfitting : {'✅ sí' if diag['overfitting'] else '❌ no'}",
        f"Underfitting: {'⚠️  posible' if diag['underfitting'] else '❌ no'}",
        f"Convergencia: {'✅ sí' if diag['converged'] else '⚠️  no'}",
        "",
        "Umbrales:",
        f"  gap > {OVERFIT_GAP_THRESHOLD}  |  rise > {OVERFIT_RISE_PCT}%",
        f"  train > {UNDERFIT_MIN_LOSS}  |  drop < {UNDERFIT_DROP_PCT}%",
    ]
    ax.text(0.03, 0.97, "\n".join(lines),
            ha="left", va="top", transform=ax.transAxes,
            family="monospace", fontsize=9)


def _draw_config_text(ax, config, diag) -> None:
    ax.axis("off")
    ax.set_title("Configuración del entrenamiento", fontsize=11, loc="left")

    es = config.get("early_stopped")
    es_str = "—" if es is None else ("sí" if es else "no")

    lvals = [
        _fmt(config.get("version")),
        _fmt(config.get("epochs")),
        _fmt(config.get("batch_size")),
        _fmt(config.get("lr"), ".3g"),
        _fmt(config.get("final_learning_rate"), ".3g"),
        _fmt(config.get("seed")),
    ]
    mvals = [
        _fmt(config.get("weight_decay"), ".2g"),
        _fmt(config.get("scheduler_patience")),
        _fmt(config.get("scheduler_factor"), ".2g"),
        _fmt(config.get("scheduler_threshold"), ".2g"),
        _fmt(config.get("min_lr"), ".2g"),
        _fmt(config.get("early_stopping_patience")),
    ]
    rvals = [
        _fmt(diag.get("num_epochs")),
        es_str,
        _fmt(config.get("stopped_epoch")),
        _duration_str(config.get("total_duration_sec")),
        _fmt(config.get("device")),
        _fmt(config.get("run_tag")),
    ]

    blocks = (
        (0.00, ["Dataset       :", "Épocas cfg    :", "Batch size    :",
                "LR inicial    :", "LR final      :", "Seed          :"], lvals),
        (0.34, ["Weight decay  :", "Sched patience:", "Sched factor  :",
                "Sched thresh  :", "Sched min_lr  :", "Early stop pat:"], mvals),
        (0.68, ["Épocas reales :", "Early stopped :", "Stop epoch    :",
                "Tiempo total  :", "Device        :", "Run tag       :"], rvals),
    )
    for x, labels, vals in blocks:
        text = "\n".join(f"{lab} {val}" for lab, val in zip(labels, vals))
        ax.text(x, 1.0, text, ha="left", va="top", transform=ax.transAxes,
                family="monospace", fontsize=9)


# ---------------------------------------------------------------------------
# Análisis de un run
# ---------------------------------------------------------------------------

def analyze_single_run(
    run_dir: Path,
    out_dir: Path,
    *,
    clean: bool = False,
    summary_dir: Path | None = None,
) -> dict:
    run_dir = Path(run_dir); out_dir = Path(out_dir)
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy = run_dir / "analysis"
    if legacy.exists() and legacy.resolve() != out_dir.resolve():
        print(f"[i] Análisis antiguo en {legacy} (no se toca).")

    metrics = get_metrics(load_history(run_dir))
    diag    = detect_patterns(metrics)
    config  = get_config(run_dir)

    # --- Figuras individuales ---
    for fname, drawer in (
        ("01_loss.png",               lambda ax: _draw_loss(ax, metrics, diag)),
        ("02_position_error.png",     lambda ax: _draw_position(ax, metrics)),
        ("03_orientation_error.png",  lambda ax: _draw_orientation(ax, metrics)),
        ("04_learning_rate.png",      lambda ax: _draw_lr(ax, metrics, config)),
        ("05_generalization_gap.png", lambda ax: _draw_gap(ax, metrics)),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        if drawer(ax):
            fig.tight_layout(); fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)

    # --- Dashboard 2x3 + franja de configuración ---
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1, 1, 0.45],
        hspace=0.35, wspace=0.25,
        left=0.04, right=0.98, top=0.94, bottom=0.03,
    )
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_gap  = fig.add_subplot(gs[0, 1])
    ax_lr   = fig.add_subplot(gs[0, 2])
    ax_pos  = fig.add_subplot(gs[1, 0])
    ax_ori  = fig.add_subplot(gs[1, 1])
    ax_stat = fig.add_subplot(gs[1, 2])
    ax_cfg  = fig.add_subplot(gs[2, :])

    fig.suptitle(
        f"Resumen del entrenamiento: {run_dir.name}",
        fontsize=14, fontweight="bold", y=0.985,
    )
    _draw_loss(ax_loss, metrics, diag)
    _draw_gap(ax_gap, metrics)
    _draw_lr(ax_lr, metrics, config)
    _draw_position(ax_pos, metrics)
    _draw_orientation(ax_ori, metrics)
    _draw_status_text(ax_stat, diag, metrics)
    _draw_config_text(ax_cfg, config, diag)

    summary_local = out_dir / "06_summary.png"
    fig.savefig(summary_local, dpi=130)
    plt.close(fig)

    # --- Copia centralizada de resúmenes ---
    if summary_dir is not None:
        summary_dir = Path(summary_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_copy = summary_dir / f"{run_dir.name}.png"
        shutil.copy2(summary_local, summary_copy)
        print(f"    summary → {summary_copy}")

    # --- Informe ---
    (out_dir / "report.md").write_text(
        _report(run_dir, diag, metrics, config), encoding="utf-8"
    )

    written = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    print(f"[✓] {run_dir.name}")
    print(f"    → {out_dir}")
    print(f"    archivos: {', '.join(written)}")
    return diag


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def _report(run_dir: Path, diag: dict, metrics: dict, config: dict) -> str:
    ts = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Informe de entrenamiento",
        "",
        f"**Generado:** `{ts}`",
        f"**Run:** `{run_dir}`",
        "",
        "📊 Dashboard visual: [`06_summary.png`](./06_summary.png)",
        "",
        "## Umbrales de diagnóstico usados",
        "| Constante | Valor | Uso |",
        "|---|---:|---|",
        f"| `OVERFIT_GAP_THRESHOLD` | {OVERFIT_GAP_THRESHOLD} | gap (val−train) > esto → overfit |",
        f"| `OVERFIT_RISE_PCT` | {OVERFIT_RISE_PCT}% | subida de val tras el mínimo |",
        f"| `UNDERFIT_MIN_LOSS` | {UNDERFIT_MIN_LOSS} | train_loss final > esto → underfit |",
        f"| `UNDERFIT_DROP_PCT` | {UNDERFIT_DROP_PCT}% | bajada mínima de train |",
        f"| `CONVERGED_SLOPE_EPS` | {CONVERGED_SLOPE_EPS} | \\|slope\\| de la cola |",
        f"| `CONVERGED_CV_EPS` | {CONVERGED_CV_EPS} | CV (sd/mu) de la cola |",
        "",
        "## Configuración",
        "| Campo | Valor |",
        "|---|---|",
        f"| Dataset | {_fmt(config.get('version'))} |",
        f"| Épocas configuradas | {_fmt(config.get('epochs'))} |",
        f"| Batch size | {_fmt(config.get('batch_size'))} |",
        f"| LR inicial | {_fmt(config.get('lr'), '.3g')} |",
        f"| LR final | {_fmt(config.get('final_learning_rate'), '.3g')} |",
        f"| Seed | {_fmt(config.get('seed'))} |",
        f"| Weight decay | {_fmt(config.get('weight_decay'), '.2g')} |",
        f"| Scheduler patience | {_fmt(config.get('scheduler_patience'))} |",
        f"| Scheduler factor | {_fmt(config.get('scheduler_factor'), '.2g')} |",
        f"| Scheduler threshold | {_fmt(config.get('scheduler_threshold'), '.2g')} |",
        f"| Scheduler min_lr | {_fmt(config.get('min_lr'), '.2g')} |",
        f"| Early stopping patience | {_fmt(config.get('early_stopping_patience'))} |",
        f"| Épocas completadas | "
        f"{_fmt(config.get('epochs_completed') or diag.get('num_epochs'))} |",
        f"| Early stopped | "
        f"{'—' if config.get('early_stopped') is None else ('sí' if config.get('early_stopped') else 'no')} |",
        f"| Stop epoch | {_fmt(config.get('stopped_epoch'))} |",
        f"| Tiempo total | {_duration_str(config.get('total_duration_sec'))} |",
        f"| Device | {_fmt(config.get('device'))} |",
        "",
    ]
    if "error" in diag:
        lines.append(f"⚠️  {diag['error']}")
        return "\n".join(lines) + "\n"

    lines += [
        "## Resumen de loss",
        f"- Épocas registradas: **{diag['num_epochs']}**",
        f"- Mejor `val_loss`: **{diag['best_val_loss']:.6f}** "
        f"(época **{diag['best_val_loss_epoch']}**)",
        f"- `val_loss` final: **{diag['final_val_loss']:.6f}**",
        f"- `train_loss` final: **{diag['final_train_loss']:.6f}**",
        f"- Gap final: **{diag['generalization_gap_final']:.6f}**",
        f"- Gap máximo: **{diag['generalization_gap_max']:.6f}**",
    ]
    if diag["val_loss_rise_pct"] is not None:
        lines.append(
            f"- Subida de `val_loss` tras la mejor época: "
            f"**{diag['val_loss_rise_pct']:.2f}%**"
        )
    if diag.get("final_slope_val_loss") is not None:
        lines.append(
            f"- Pendiente de `val_loss` en la cola: "
            f"**{diag['final_slope_val_loss']:+.5f}/época**"
        )
    if diag.get("train_loss_drop_pct") is not None:
        lines.append(
            f"- Bajada de `train_loss` desde la época 1: "
            f"**{diag['train_loss_drop_pct']:.2f}%**"
        )

    lines += ["", "## Diagnóstico heurístico"]
    lines.append(f"- Overfitting: {'✅ sí' if diag['overfitting'] else '❌ no detectado'}")
    lines.append(f"- Underfitting: {'⚠️  posible' if diag['underfitting'] else '❌ no detectado'}")
    lines.append(f"- Convergencia: {'✅ sí' if diag['converged'] else '⚠️  aún con cambios'}")

    for title, key, unit in (
        ("Error de posición (val)",    "val_pos_mean", "m"),
        ("Error de orientación (val)", "val_ori_mean", "°"),
    ):
        arr = _safe(metrics.get(key))
        if arr is None or arr.size == 0 or np.all(np.isnan(arr)):
            continue
        i = int(np.nanargmin(arr))
        lines += ["", f"## {title}"]
        lines.append(f"- Mejor error medio: **{arr[i]:.4f} {unit}** (época {i + 1})")
        lines.append(f"- Error medio final: **{arr[-1]:.4f} {unit}**")

    lines.append("\n---\n*Los diagnósticos son heurísticos; úsalos como guía.*")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Selección de runs y salida
# ---------------------------------------------------------------------------

def _collect_targets(parser, args) -> list[Path]:
    targets: list[Path] = []

    for raw in args.paths:
        path = Path(raw)
        if (path / "history.json").is_file():
            targets.append(path); continue
        runs = discover_runs(path)
        if not runs:
            parser.error(f"'{path}' no contiene history.json ni subdirectorios con él.")
        if len(runs) == 1:
            targets.append(runs[0])
        else:
            parser.error(
                f"'{path}' contiene {len(runs)} runs. "
                f"Indica uno con --run <run_dir> o usa --all {path} "
                f"(o --latest {path} para el más reciente)."
            )

    for raw in args.run:
        path = Path(raw)
        if not (path / "history.json").is_file():
            parser.error(f"--run '{path}' no contiene history.json.")
        targets.append(path)

    if args.latest:
        latest = find_latest_run(args.latest)
        if latest is None:
            parser.error(f"No se encontró ningún run bajo '{args.latest}'.")
        print(f"[i] Run más reciente bajo {args.latest}: {latest}")
        targets.append(latest)

    if args.all:
        runs = discover_runs(args.all)
        if not runs:
            parser.error(f"No se encontró ningún run bajo '{args.all}'.")
        print(f"[i] Analizando TODOS los runs bajo {args.all} ({len(runs)})")
        targets.extend(runs)

    seen, unique = set(), []
    for t in targets:
        k = t.resolve()
        if k not in seen:
            seen.add(k); unique.append(t)
    return unique


def _resolve_out_dir(run_dir, *, out_override, out_base, shared_root) -> Path:
    if out_override is not None:
        return out_override
    if out_base is not None:
        return out_base / run_dir.name
    if shared_root is not None:
        return shared_root / "analysis" / "runs" / run_dir.name
    return run_dir.parent / "analysis" / "runs" / run_dir.name


def _resolve_summary_dir(run_dir, *, explicit, out_base, shared_root) -> Path:
    if explicit is not None:
        return explicit
    if out_base is not None:
        return out_base / "summaries"
    base = shared_root if shared_root is not None else run_dir.parent
    return base / "analysis" / "summaries"


def _resolve_reports_file(*, first_run: Path, out_base, shared_root, explicit) -> Path:
    """Ruta del .md consolidado con todos los report.md."""
    if explicit is not None:
        return Path(explicit)
    if out_base is not None:
        return out_base / "all_reports.md"
    if shared_root is not None:
        return shared_root / "analysis" / "all_reports.md"
    return first_run.parent / "analysis" / "all_reports.md"


def _write_consolidated_reports(path: Path, reports: list[tuple[str, str]]) -> None:
    """Concatena los reportes de cada run en un único .md con índice."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _anchor(name: str) -> str:
        return name.lower().replace("_", "-").replace(".", "-").replace(" ", "-")

    ts = datetime.now().isoformat(timespec="seconds")
    parts: list[str] = [
        "# Reportes consolidados de entrenamientos\n",
        f"**Generado:** `{ts}`  ",
        f"**Total de runs:** {len(reports)}\n",
        "> Cada run vive también en su carpeta `analysis/runs/<run>/`. "
        "Los enlaces relativos a `06_summary.png` dentro de cada bloque "
        "solo funcionan abriendo el `report.md` de esa carpeta.\n",
        "## Índice\n",
    ]
    for i, (name, _text) in enumerate(reports, 1):
        parts.append(f"{i}. [`{name}`](#{_anchor(name)})")
    parts.append("\n---")

    for name, text in reports:
        parts.append(f'\n\n<a name="{_anchor(name)}"></a>')
        parts.append(f"\n## 🏃 `{name}`\n")
        # Demote headings by 2 so they fit under the H2 banner
        for line in text.split("\n"):
            parts.append("##" + line if line.startswith("#") else line)
        parts.append("\n---")

    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza entrenamientos con dashboard y configuración.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python -m src.analysis_nn.analyze_run --all outputs_nn/resnet18\n"
            "  python -m src.analysis_nn.analyze_run --latest outputs_nn/resnet18\n"
            "  python -m src.analysis_nn.analyze_run outputs_nn/resnet18/v2_0_run_05\n"
            "  python -m src.analysis_nn.analyze_run --all outputs_nn/resnet18 --clean\n"
            "  python -m src.analysis_nn.analyze_run --all outputs_nn/resnet18 "
            "--reports-file resumen.md\n"
        ),
    )
    parser.add_argument("paths", nargs="*",
                        help="Rutas de runs individuales (con history.json).")
    parser.add_argument("--run", action="append", default=[], metavar="RUN_DIR",
                        help="Run concreto (repetible).")
    parser.add_argument("--latest", metavar="ROOT",
                        help="Solo el run más reciente bajo ROOT.")
    parser.add_argument("--all", metavar="ROOT",
                        help="Todos los runs bajo ROOT.")
    parser.add_argument("--out-base", metavar="DIR", default=None,
                        help="Carpeta base de análisis (una subcarpeta por run).")
    parser.add_argument("--out", metavar="DIR", default=None,
                        help="Ruta de salida (solo si hay 1 run).")
    parser.add_argument("--summary-dir", metavar="DIR", default=None,
                        help="Carpeta centralizada con los 06_summary.png. "
                             "Por defecto: <root>/analysis/summaries.")
    parser.add_argument("--reports-file", metavar="PATH", default=None,
                        help="Ruta del .md consolidado con los reportes de todos "
                             "los runs. Por defecto: <analysis>/all_reports.md.")
    parser.add_argument("--clean", action="store_true",
                        help="Borra la subcarpeta de cada run antes de escribir.")
    args = parser.parse_args()

    targets = _collect_targets(parser, args)
    if not targets:
        parser.error("No se indicó ningún run. Usa <run_dir>, --run, --latest o --all.")

    single_out = Path(args.out) if args.out else None
    if single_out and len(targets) != 1:
        parser.error("--out solo es válido cuando se analiza un único run.")
    out_base    = Path(args.out_base) if args.out_base else None
    shared_root = Path(args.all or args.latest) if (args.all or args.latest) else None
    explicit_summary_dir = Path(args.summary_dir) if args.summary_dir else None

    ok, fail = 0, 0
    reports_accumulated: list[tuple[str, str]] = []

    for run in targets:
        try:
            out_dir = _resolve_out_dir(
                run, out_override=single_out, out_base=out_base, shared_root=shared_root,
            )
            summary_dir = _resolve_summary_dir(
                run, explicit=explicit_summary_dir,
                out_base=out_base, shared_root=shared_root,
            )
            analyze_single_run(
                run, out_dir, clean=args.clean, summary_dir=summary_dir,
            )
            # Leer el report recién escrito para el consolidado
            report_md = out_dir / "report.md"
            if report_md.is_file():
                reports_accumulated.append(
                    (run.name, report_md.read_text(encoding="utf-8"))
                )
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"[!] Error en {run}: {exc}")

    # --- Consolidado de reportes ---
    reports_path: Path | None = None
    if reports_accumulated:
        reports_path = _resolve_reports_file(
            first_run=targets[0],
            out_base=out_base,
            shared_root=shared_root,
            explicit=Path(args.reports_file) if args.reports_file else None,
        )
        _write_consolidated_reports(reports_path, reports_accumulated)

    # --- Resumen final ---
    print(f"\nResumen: {ok} análisis generados, {fail} con error.")
    if reports_path is not None:
        print(f"Reportes consolidados ({len(reports_accumulated)} runs) → {reports_path}")
    if explicit_summary_dir is not None:
        print(f"Resúmenes (carpeta): {explicit_summary_dir}")
    elif shared_root is not None:
        print(f"Resúmenes (carpeta): {shared_root / 'analysis' / 'summaries'}")
    elif out_base is not None:
        print(f"Resúmenes (carpeta): {out_base / 'summaries'}")


if __name__ == "__main__":
    main()