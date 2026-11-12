# Comparación de runs

Runs comparados: **11**

## Umbrales de diagnóstico

| Constante | Valor | Uso |
|---|---:|---|
| `OVERFIT_GAP_THRESHOLD` | 0.05 | gap (val−train) > esto → overfit |
| `OVERFIT_RISE_PCT` | 5.0% | subida de val tras el mínimo para confirmar overfit |
| `UNDERFIT_MIN_LOSS` | 0.2 | train_loss final > esto → posible underfit |
| `UNDERFIT_DROP_PCT` | 10.0% | bajada mínima de train para no ser underfit |
| `CONVERGED_SLOPE_EPS` | 0.001 | |slope|(cola) < esto → convergido |
| `CONVERGED_CV_EPS` | 0.02 | coef. variación (sd/mu) de la cola < esto → convergido |

> Fuente única de verdad: `src/analysis_nn/__init__.py`. Cambiar allí y todo el pipeline queda consistente.

## Tabla resumen

| Run | Épocas | Tiempo | Mejor val_loss | Ép. | Mejor pos (m) | Mejor ori (°) | Gap final | Overfit | Underfit | Conv. |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| v2_0_run_03_ep80_bs4_lr0p0001 | 80 | 3h 22m | 0.135746 | 79 | 0.3858 | 9.6622 | 0.1672 | sí | no | no |
| v2_0_run_04_ep10_bs8_lr0p0001 | 10 | 23m 18s | 0.160011 | 8 | 0.4861 | 10.8822 | 0.1020 | sí | no | no |
| v2_0_run_05_ep10_bs4_lr0p0001 | 10 | 24m 59s | 0.170574 | 10 | 0.4748 | 12.4471 | 0.0655 | sí | no | no |
| v2_0_run_06_ep10_bs4_lr5e-05 | 10 | 25m 49s | 0.156697 | 10 | 0.4532 | 14.9126 | -0.0092 | no | no | no |
| v2_0_run_07_ep10_bs4_lr0p0002 | 10 | 25m 28s | 0.208692 | 7 | 0.5431 | 12.5189 | 0.1021 | sí | no | no |
| v2_0_run_08_ep80_bs8_lr0p0001 | 80 | 3h 7m | 0.118176 | 67 | 0.3771 | 8.8123 | 0.1153 | sí | no | no |
| v2_0_run_09_ep80_bs8_lr5e-05 | 80 | 3h 8m | 0.129987 | 17 | 0.4230 | 9.3251 | 0.1547 | sí | no | no |
| v2_0_run_10_ep300_bs8_lr0p0001 | 300 | 11h 42m | 0.118176 | 67 | 0.3724 | 8.8123 | 0.1738 | sí | no | sí |
| v2_0_run_11_ep300_bs8_lr0p0001_sd123 | 300 | 11h 40m | 0.120669 | 164 | 0.3857 | 8.2238 | 0.1727 | sí | no | sí |
| v2_0_run_13_ep120_bs8_lr0p0001_sd42_scheduler_ep120 | 76 | 2h 57m | 0.138419 | 26 | 0.4056 | 9.1771 | 0.1416 | sí | no | sí |
| v2_0_run_14_ep120_bs8_lr0p0001_sd42_wd0p0001_e14_wd | 116 | 4h 29m | 0.136431 | 66 | 0.4017 | 8.7804 | 0.1534 | sí | no | sí |

## Ganador por val_loss
- `v2_0_run_08_ep80_bs8_lr0p0001` con **0.1182**

## Ganador por error de posición
- `v2_0_run_10_ep300_bs8_lr0p0001` con **0.3724 m**

## Ganador por error de orientación
- `v2_0_run_11_ep300_bs8_lr0p0001_sd123` con **8.2238°**

## Tiempo de entrenamiento
- Runs con tiempo registrado: **11/11**.
- Más rápido: `v2_0_run_04_ep10_bs8_lr0p0001` con **23m 18s**.
- Más lento:  `v2_0_run_10_ep300_bs8_lr0p0001` con **11h 42m**.
- Media: **3h 49m**.

## Figuras generadas

- `01_val_loss_compare.png` — Curvas val_loss (genérica, escala log + zoom)
- `02_best_val_loss.png` — Ranking por mejor val_loss
- `03_best_position_error.png` — Ranking por mejor error de posición
- `04_best_orientation_error.png` — Ranking por mejor error de orientación
- `05_numeric_summary.md` — Los 5 números clave por run (+ tiempo)
- `06_sanity_checks.md` — Checks E08/E10 y E13/E14
- `07_fig1_pipeline_overview.png` — FIG1 — Pipeline: E01 → E03 → E08
- `08_fig2_hyperparam_sweep.png` — FIG2 — Batch (A) y LR (B)
- `09_fig3_overfitting.png` — FIG3 — Sobreajuste E08 vs E10 + línea límite
- `10_fig4_control_mechanisms.png` — FIG4 — E13 (scheduler+early) y E14 (+wd)
- `11_fig5_tradeoff_scatter.png` — FIG5 — Trade-off val_loss vs posición
- `12_fig5b_overfit_diagnostic.png` — FIG5b — Diagnóstico por run con líneas límite
- `13_fig6_pair_compare.png` — FIG6 — Comparación de 2 runs con umbral overfit
- `14_fig7_training_times.png` — FIG7 — Tiempos de ejecución (total y por época)