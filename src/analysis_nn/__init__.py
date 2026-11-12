"""Utilidades compartidas para el análisis de entrenamientos ResNet-18."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Umbrales de diagnóstico (fuente única de verdad)
# ---------------------------------------------------------------------------
# Estos valores los usan:
#   - detect_patterns()  → para asignar tags OVERFIT / UNDERFIT / NO CONV.
#   - compare_runs.py    → para dibujar líneas de referencia coherentes.
#   - analyze_run.py     → para el report.md.
# Cambiar aquí y todo el pipeline queda consistente.

OVERFIT_GAP_THRESHOLD = 0.05   # gap final (val − train) > esto → overfit
OVERFIT_RISE_PCT      = 5.0    # % de subida de val_loss tras el mejor epoch
UNDERFIT_MIN_LOSS     = 0.2    # train_loss final > esto → posible underfit
UNDERFIT_DROP_PCT     = 10.0   # bajada % de train_loss desde el epoch 1
CONVERGED_SLOPE_EPS   = 1e-3   # |pendiente| de val_loss en la cola < esto
CONVERGED_CV_EPS      = 0.02   # coef. variación (sd/mu) de la cola < esto


# ---------------------------------------------------------------------------
# Carga y normalización de history.json
# ---------------------------------------------------------------------------

def load_history(run_dir: Path | str) -> dict[str, list]:
    """Carga `history.json` y lo normaliza a un dict de listas.

    Admite los dos formatos habituales:
      1) dict de listas:  {"train_loss": [...], "val_loss": [...]}
      2) lista de dicts:  [{"train_loss": 1.0, "val_loss": 1.2}, ...]
    También acepta que la lista venga anidada bajo la clave "history".
    """
    run_dir = Path(run_dir)
    hist_path = run_dir / "history.json"
    if not hist_path.is_file():
        raise FileNotFoundError(f"No existe {hist_path}")

    with hist_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return _normalize_list(raw)

    if isinstance(raw, dict):
        if isinstance(raw.get("history"), list):
            return _normalize_list(raw["history"])
        return {
            k: (list(v) if isinstance(v, list) else [v])
            for k, v in raw.items()
            if isinstance(v, (list, int, float))
        }

    raise ValueError(f"Formato de history.json no reconocido: {hist_path}")


def _normalize_list(rows: list[dict]) -> dict[str, list]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    return {k: [row.get(k) for row in rows] for k in keys}


def find_metric(history: dict, *candidates: str) -> list | None:
    """Devuelve la primera lista que coincida con alguno de los nombres."""
    for name in candidates:
        if name in history and isinstance(history[name], list):
            return history[name]
    return None


def get_metrics(history: dict) -> dict[str, list | None]:
    """Extrae métricas relevantes probando nombres alternativos.

    Convención de los scripts actuales (train.py / train_2.py):
      - "train_loss"                       -> loss de train
      - "loss"                             -> loss de VALIDACIÓN
      - "position_error_mean" / "_median"  -> VALIDACIÓN (metros)
      - "orientation_error_mean_deg" / ...
                                           -> VALIDACIÓN (grados)
      - "learning_rate"                    -> LR de la época (solo train_2.py)
    """
    metrics: dict[str, list | None] = {
        "train_loss": find_metric(history, "train_loss", "loss_train", "train/loss"),
        "val_loss": find_metric(
            history,
            "val_loss", "loss_val", "validation_loss", "val/loss",
            "loss",          # <-- fallback: en train_2.py "loss" ya es val_loss
        ),
        "lr": find_metric(history, "lr", "learning_rate", "train/lr"),
    }

    metrics["train_pos_mean"] = find_metric(
        history,
        "train_pos_error_mean", "train_position_error_mean",
        "train_pos_mean", "train_position_mean",
        "train_mean_position_error",
    )
    metrics["train_pos_median"] = find_metric(
        history,
        "train_pos_error_median", "train_position_error_median",
        "train_pos_median", "train_position_median",
        "train_median_position_error",
    )
    metrics["train_ori_mean"] = find_metric(
        history,
        "train_angle_error_mean", "train_orientation_error_mean",
        "train_ori_mean", "train_orientation_mean",
        "train_mean_angle_error", "train_mean_angular_error",
    )
    metrics["train_ori_median"] = find_metric(
        history,
        "train_angle_error_median", "train_orientation_error_median",
        "train_ori_median", "train_orientation_median",
        "train_median_angle_error", "train_median_angular_error",
    )

    metrics["val_pos_mean"] = find_metric(
        history,
        "val_pos_error_mean", "val_position_error_mean",
        "val_pos_mean", "val_position_mean",
        "val_mean_position_error",
        "position_error_mean",
    )
    metrics["val_pos_median"] = find_metric(
        history,
        "val_pos_error_median", "val_position_error_median",
        "val_pos_median", "val_position_median",
        "val_median_position_error",
        "position_error_median",
    )
    metrics["val_ori_mean"] = find_metric(
        history,
        "val_angle_error_mean", "val_orientation_error_mean",
        "val_ori_mean", "val_orientation_mean",
        "val_mean_angle_error", "val_mean_angular_error",
        "orientation_error_mean_deg",
        "orientation_error_mean",
    )
    metrics["val_ori_median"] = find_metric(
        history,
        "val_angle_error_median", "val_orientation_error_median",
        "val_ori_median", "val_orientation_median",
        "val_median_angle_error", "val_median_angular_error",
        "orientation_error_median_deg",
        "orientation_error_median",
    )

    return metrics


# ---------------------------------------------------------------------------
# Descubrimiento de runs
# ---------------------------------------------------------------------------

def discover_runs(root: Path | str) -> list[Path]:
    """Devuelve todos los directorios que contienen un history.json."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("history.json")})


def find_latest_run(root: Path | str) -> Path | None:
    """Devuelve el run con `history.json` más reciente bajo `root` (por mtime)."""
    runs = discover_runs(root)
    if not runs:
        return None
    return max(runs, key=lambda p: (p / "history.json").stat().st_mtime)


def find_latest_runs(root: Path | str, n: int) -> list[Path]:
    """Devuelve los N runs con `history.json` más recientes bajo `root`."""
    runs = discover_runs(root)
    runs.sort(key=lambda p: (p / "history.json").stat().st_mtime, reverse=True)
    return runs[:n]


# ---------------------------------------------------------------------------
# Diagnóstico heurístico
# ---------------------------------------------------------------------------

def detect_patterns(metrics: dict[str, list | None]) -> dict:
    """Analiza las métricas y devuelve diagnósticos interpretables.

    Usa las constantes de umbral definidas arriba. Todos los tags son
    heurísticos y combinables (un run puede ser OVERFIT y no converger
    simultáneamente); `compare_runs.py` decide la etiqueta principal
    con prioridad OVERFIT > UNDERFIT > NO CONV. > OK.
    """
    train, val = metrics.get("train_loss"), metrics.get("val_loss")
    diag: dict = {}

    if not train or not val:
        diag["error"] = "Faltan train_loss o val_loss en history.json"
        return diag

    train_arr = np.asarray([np.nan if v is None else v for v in train], dtype=float)
    val_arr   = np.asarray([np.nan if v is None else v for v in val],   dtype=float)

    if np.all(np.isnan(val_arr)):
        diag["error"] = "val_loss contiene solo NaN"
        return diag

    n = len(val_arr)
    best_idx = int(np.nanargmin(val_arr))
    best_val = float(val_arr[best_idx])
    final_val = float(val_arr[-1])

    diag["num_epochs"]          = n
    diag["best_val_loss"]       = best_val
    diag["best_val_loss_epoch"] = best_idx + 1
    diag["final_val_loss"]      = final_val

    gap = val_arr - train_arr
    gap_final = float(gap[-1])
    diag["generalization_gap_final"] = gap_final
    diag["generalization_gap_max"]   = float(np.nanmax(gap))

    # --- Overfitting: gap final grande O subida significativa de val_loss ---
    if best_idx < n - 1 and best_val > 0:
        rise     = final_val - best_val
        rise_pct = 100.0 * rise / best_val
    else:
        rise     = 0.0
        rise_pct = 0.0
    diag["val_loss_rise_after_best"] = float(rise)
    diag["val_loss_rise_pct"]        = float(rise_pct)
    diag["overfitting"] = bool(
        rise_pct > OVERFIT_RISE_PCT or gap_final > OVERFIT_GAP_THRESHOLD
    )

    # --- Underfitting: train_loss final alto Y apenas bajó desde el epoch 1 ---
    initial_train = float(train_arr[0])
    final_train   = float(train_arr[-1]) if not np.isnan(train_arr[-1]) else initial_train
    if np.isfinite(initial_train) and initial_train > 0:
        drop_pct = 100.0 * (initial_train - final_train) / initial_train
    else:
        drop_pct = None
    diag["train_loss_drop_pct"] = float(drop_pct) if drop_pct is not None else None
    diag["final_train_loss"]    = final_train
    diag["underfitting"] = bool(
        final_train > UNDERFIT_MIN_LOSS
        and (drop_pct is None or drop_pct < UNDERFIT_DROP_PCT)
    )

    # --- Convergencia: cola estable → |slope| pequeño O CV bajo ---
    tail = max(3, n // 10)
    last = val_arr[-tail:]
    mask = ~np.isnan(last)
    if mask.sum() >= 2:
        x = np.arange(len(last))[mask]
        y = last[mask]
        slope = float(np.polyfit(x, y, 1)[0])
        mu, sd = float(np.mean(y)), float(np.std(y))
        cv = (sd / mu) if mu > 0 else np.inf
        diag["final_slope_val_loss"] = slope
        diag["tail_cv"]              = float(cv)
        diag["converged"] = bool(
            abs(slope) < CONVERGED_SLOPE_EPS or cv < CONVERGED_CV_EPS
        )
    else:
        diag["final_slope_val_loss"] = None
        diag["tail_cv"]              = None
        diag["converged"]            = False

    return diag


# ---------------------------------------------------------------------------
# run_info.json + nombre de carpeta
# ---------------------------------------------------------------------------

def load_run_info(run_dir: Path | str) -> dict:
    """Carga run_info.json si existe; devuelve {} si no."""
    p = Path(run_dir) / "run_info.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


_RUN_NAME_RE = re.compile(
    r"^(?P<version>.+?)_run_(?P<run>\d+)_ep(?P<epochs>\d+)_bs(?P<bs>\d+)"
    r"_lr(?P<lr>[\dp.]+)(?:_sd(?P<seed>\d+))?"
    r"(?:_wd(?P<wd>[\dp.]+))?"
    r"(?:_(?P<tag>.+))?$"
)


def parse_run_name(run_dir: Path | str) -> dict:
    """Extrae parámetros del nombre de la carpeta de run (fallback)."""
    name = Path(run_dir).name
    m = _RUN_NAME_RE.match(name)
    if not m:
        return {}

    def _to_float(s):
        return float(s.replace("p", ".")) if s else None

    return {
        "version":      m.group("version"),
        "run_number":   int(m.group("run")),
        "epochs":       int(m.group("epochs")),
        "batch_size":   int(m.group("bs")),
        "lr":           _to_float(m.group("lr")),
        "seed":         int(m.group("seed")) if m.group("seed") else None,
        "weight_decay": _to_float(m.group("wd")),
        "run_tag":      m.group("tag"),
    }


def get_config(run_dir: Path | str) -> dict:
    """Mezcla run_info.json → args → nombre de carpeta en un único dict.

    Prioridad: args de run_info.json > otros campos de run_info.json >
    nombre de carpeta. `final_learning_rate`, `total_duration_sec`,
    `epochs_completed`, `early_stopped`, `stopped_epoch` y `device` solo
    existen en run_info.json.
    """
    info = load_run_info(run_dir)
    args = info.get("args", {}) if isinstance(info.get("args"), dict) else {}
    parsed = parse_run_name(run_dir)

    def pick(*keys):
        for source in (args, info, parsed):
            for k in keys:
                if k in source and source[k] is not None:
                    return source[k]
        return None

    return {
        "version":                 pick("version", "dataset_version"),
        "epochs":                  pick("epochs"),
        "batch_size":              pick("batch_size"),
        "lr":                      pick("lr"),
        "seed":                    pick("seed"),
        "weight_decay":            pick("weight_decay"),
        "scheduler_patience":      pick("scheduler_patience"),
        "scheduler_factor":        pick("scheduler_factor"),
        "scheduler_threshold":     pick("scheduler_threshold"),
        "min_lr":                  pick("min_lr", "scheduler_min_lr"),
        "early_stopping_patience": pick("early_stopping_patience"),
        "run_tag":                 pick("run_tag"),
        "device":                  info.get("device"),
        "start_time":              info.get("start_time"),
        "end_time":                info.get("end_time"),
        "total_duration_sec":      info.get("total_duration_sec"),
        "epochs_completed":        info.get("epochs_completed"),
        "early_stopped":           info.get("early_stopped"),
        "stopped_epoch":           info.get("stopped_epoch"),
        "final_learning_rate":     info.get("final_learning_rate"),
        "best_val_loss_from_info": info.get("best_val_loss"),
    }