# Extracción numérica por run

Los 5 números clave de cada curva (+ tiempo total):

| Run | Épocas | Tiempo | Mejor val_loss | Ép. | Mejor pos (m) | Ép. | Mejor ori (°) | Ép. | Gap final | Pendiente val (últ. 20) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_0_run_03_ep80_bs4_lr0p0001 | 80 | 3h 22m | 0.135746 | 79 | 0.3858 | 79 | 9.6622 | 79 | 0.1672 | -0.00342 |
| v2_0_run_04_ep10_bs8_lr0p0001 | 10 | 23m 18s | 0.160011 | 8 | 0.4861 | 6 | 10.8822 | 10 | 0.1020 | -0.00920 |
| v2_0_run_05_ep10_bs4_lr0p0001 | 10 | 24m 59s | 0.170574 | 10 | 0.4748 | 10 | 12.4471 | 10 | 0.0655 | -0.01699 |
| v2_0_run_06_ep10_bs4_lr5e-05 | 10 | 25m 49s | 0.156697 | 10 | 0.4532 | 10 | 14.9126 | 10 | -0.0092 | -0.02726 |
| v2_0_run_07_ep10_bs4_lr0p0002 | 10 | 25m 28s | 0.208692 | 7 | 0.5431 | 7 | 12.5189 | 9 | 0.1021 | -0.01619 |
| v2_0_run_08_ep80_bs8_lr0p0001 | 80 | 3h 7m | 0.118176 | 67 | 0.3771 | 67 | 8.8123 | 74 | 0.1153 | -0.00059 |
| v2_0_run_09_ep80_bs8_lr5e-05 | 80 | 3h 8m | 0.129987 | 17 | 0.4230 | 33 | 9.3251 | 79 | 0.1547 | -0.00039 |
| v2_0_run_10_ep300_bs8_lr0p0001 | 300 | 11h 42m | 0.118176 | 67 | 0.3724 | 164 | 8.8123 | 74 | 0.1738 | -0.00065 |
| v2_0_run_11_ep300_bs8_lr0p0001_sd123 | 300 | 11h 40m | 0.120669 | 164 | 0.3857 | 164 | 8.2238 | 279 | 0.1727 | +0.00084 |
| v2_0_run_13_ep120_bs8_lr0p0001_sd42_scheduler_ep120 | 76 | 2h 57m | 0.138419 | 26 | 0.4056 | 66 | 9.1771 | 60 | 0.1416 | -0.00003 |
| v2_0_run_14_ep120_bs8_lr0p0001_sd42_wd0p0001_e14_wd | 116 | 4h 29m | 0.136431 | 66 | 0.4017 | 66 | 8.7804 | 105 | 0.1534 | +0.00017 |

**Umbrales usados para clasificar (fuente: `src/analysis_nn/__init__.py`):**

- `OVERFIT_GAP_THRESHOLD = 0.05`
- `OVERFIT_RISE_PCT = 5.0`
- `UNDERFIT_MIN_LOSS = 0.2`
- `UNDERFIT_DROP_PCT = 10.0`
- `CONVERGED_SLOPE_EPS = 0.001`
- `CONVERGED_CV_EPS = 0.02`

**Lectura:**

- `Gap final` = `val_loss` final − `train_loss` final. Positivo grande → sobreajuste.
- `Pendiente val` > 0 en las últimas 20 épocas → el modelo ya no generaliza; < 0 → aún aprendía.
- `Ép.` junto a cada métrica indica si el mejor valor coincide con el mejor `val_loss` o no.
- `Tiempo` = tiempo total de entrenamiento extraído del history o de `metadata.json` (vacío si no está registrado).