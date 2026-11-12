# Technical Advances — Historial técnico del proyecto

**Proyecto:** `drone-6dof-pose-estimation` · **Autor:** Ricardo Ramos Morales · **Asesor:** Dr. José Martínez Carranza · **INAOE**
**Periodo:** 1 sep – 27 nov 2026 · **Última actualización:** 11 de noviembre de 2026

> Registro técnico acumulativo. Las secciones nuevas se añaden al final de cada bloque y las entradas no se borran; si algo cambia, se anota la corrección con fecha.

---

## 1. Objetivo técnico

Estimar `[x, y, z, qx, qy, qz, qw]` de la cámara del Bebop 2 a partir de una imagen, en metros y marco FLU (X adelante, Y izquierda, Z arriba), sin odometría ni sensores externos en inferencia.

---

## 2. Dataset v2_0

### 2.1 Contenido
| Elemento | Valor |
|---|---|
| Videos | 5 recorridos, 856 × 480 px, ~30 FPS |
| Extracción | ~2 FPS con recortes por video |
| Frames de recorridos | 1,299 |
| Fotos de referencia | 15 (posición física medida) |
| Total | 1,314 (`frames_all`) |
| Registradas por COLMAP | 1,189 (≈ 90.5 %) |

| Secuencia | Extraídos | Registrados | Uso |
|---|---:|---:|---|
| `bebop_001_val` | 325 | 305 | Validación |
| `bebop_002_train` | 165 | 138 | Train |
| `bebop_003_train` | 222 | 217 | Train |
| `bebop_004_train` | 268 | 238 | Train |
| `bebop_005_train` | 319 | 276 | Train |
| Referencias | 15 | 15 | Alineación |

**Pendiente de verificar:** en `v2_0` train suma 869 por secuencias, pero un resumen previo reporta 884; la diferencia (15) coincide con el número de referencias. Confirmar los conteos reales en `poses_split.csv` (p. ej. `pd.read_csv(...)['split'].value_counts()`).

**Historial:** `v1_0` (12 videos, 646 frames, 609 registrados) se descartó porque no había distancias físicas para fijar la escala en COLMAP y el ambiente donde se capturó había cambiado. `v2_0` se capturó con 15 referencias medidas. Los 869/884 no provienen de `v1_0`.

### 2.2 Reconstrucción COLMAP
Modelo `SIMPLE_RADIAL`, 107,703 puntos 3D, error medio de reproyección 0.966 px, COLMAP 3.9.1 sin CUDA. Hubo una primera ronda (12 videos, 646 frames, 609 registrados) que se conserva como antecedente metodológico.

### 2.3 Alineación a FLU
Transformación de similitud 3D: `p_FLU = s · R · p_COLMAP + t` (Umeyama con SVD), usando las 15 referencias.

| Métrica | Valor |
|---|---:|
| Escala | 0.394834 m / unidad COLMAP |
| MAE 3D | 0.019696 m |
| RMSE 3D | 0.026765 m |
| Error relativo | 1.643 % |

**Limitación:** los residuos se calculan sobre los mismos puntos del ajuste → no es validación independiente. Falta reservar referencias fuera del ajuste.

### 2.4 Validación de poses
1,181 válidas · 8 sospechosas · 0 inválidas. Ejemplo sospechoso: `bebop_001_val_frame_00043`, posición ≈ (−15.289, −8.515, 6.094) m (salto atípico). No se ha corregido ni eliminado nada automáticamente.

---

## 3. Implementación

### 3.1 Modelo (`model.py`)
- `ResNetPoseRegressor`: ResNet-18 preentrenada (ImageNet), `fc` reemplazada por `Linear(512, 7)`.
- Salida cruda: 3 de posición (normalizada) y 4 de cuaternión; el cuaternión se normaliza (L2) antes de pérdida/métricas.

### 3.2 Pérdida
```text
L_pos = MSE( pos_pred , (pos_real − media_train) / std_train )
L_ori = mean( 1 − |q̂_pred · q̂_real| )          # invariante a q ↔ −q
L     = 1.0 · L_pos + 1.0 · L_ori
```
Los pesos son fijos (1:1). L_pos está en unidades de desviación estándar y L_ori en [0, 1], así que las dos escalas no son equivalentes; esto es relevante al comparar con variantes de PoseNet que aprenden o ajustan el balance.

### 3.3 Datos y entrenamiento
| Elemento | Valor |
|---|---|
| Entrada | Resize 224 × 224 (sin recorte, no conserva relación de aspecto 856 × 480), normalización ImageNet |
| Aumento de datos | Ninguno en `dataset.py` |
| Estadísticas de posición | Solo train: media X 1.47235, Y −1.17480, Z 0.85308; std 1.05284, 0.93395, 0.57002 |
| Optimizador | `torch.optim.Adam` (no AdamW) |
| Dispositivo | CPU (Ryzen 5 3500U, 24 GB) |
| DataLoader | `num_workers=0`, `shuffle=True` en train |

### 3.4 Métricas
- Posición: norma euclidiana en metros (media y mediana).
- Orientación: `2 · arccos(|q̂_pred · q̂_real|)` en grados.
- `train_2.py` exporta `val_predictions.csv` con el mejor checkpoint (predicción y error por imagen).

### 3.5 Scripts de entrenamiento
- `train.py`: versión base.
- `train_2.py`: añade `ReduceLROnPlateau` (patience 10, factor 0.5, threshold 1e-3, min_lr 1e-6), early stopping (patience por defecto 50), `weight_decay`, registro de LR, `run_info.json` y exportación de predicciones.

---

## 4. Experimentos ResNet-18 (dataset v2_0, Adam, CPU)

### 4.1 Tabla maestra
Los mejores errores se toman **por separado** de cualquier época; no corresponden necesariamente al mismo checkpoint.

| Run | Batch | LR | Seed | Épocas (cfg / reales) | Mejor val_loss (ép.) | Mejor pos. media (ép.) | Mejor ori. media (ép.) | Gap final | Tiempo |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| E01 | 2 | 1e-4 | 42 | 2 / 2 | 0.43314 (1) | 0.8442 m | 24.99° (1) | −0.077 | 5 m 52 s |
| E02 | 1 | 1e-4 | 42 | 4 / 4 | 3.08903 (2) | 2.2641 m | 80.27° (3) | 3.442 | 17 m 08 s |
| E03 | 4 | 1e-4 | 42 | 80 / 80 | 0.13575 (79) | 0.3858 m (79) | 9.66° (79) | 0.167 | 3 h 22 m |
| E04 | 8 | 1e-4 | 42 | 10 / 10 | 0.16001 (8) | 0.4861 m (6) | 10.88° (10) | 0.102 | 23 m 18 s |
| E05 | 4 | 1e-4 | 42 | 10 / 10 | 0.17057 (10) | 0.4748 m (10) | 12.45° (10) | 0.066 | 24 m 59 s |
| E06 | 4 | 5e-5 | 42 | 10 / 10 | 0.15670 (10) | 0.4532 m (10) | 14.91° (10) | −0.009 | 25 m 49 s |
| E07 | 4 | 2e-4 | 42 | 10 / 10 | 0.20869 (7) | 0.5431 m (7) | 12.52° (9) | 0.102 | 25 m 28 s |
| **E08** | 8 | 1e-4 | 42 | 80 / 80 | **0.11818 (67)** | 0.3771 m (67) | 8.81° (74) | 0.115 | 3 h 07 m |
| E09 | 8 | 5e-5 | 42 | 80 / 80 | 0.12999 (17) | 0.4230 m (33) | 9.33° (79) | 0.155 | 3 h 08 m |
| E10 | 8 | 1e-4 | 42 | 300 / 300 | 0.11818 (67) | **0.3724 m (164)** | 8.81° (74) | 0.174 | 11 h 42 m |
| E11 | 8 | 1e-4 | 123 | 300 / 300 | 0.12067 (164) | 0.3857 m (164) | **8.22° (279)** | 0.173 | 11 h 40 m |
| E12 | 8 | 1e-4 | 42 | 2 / 2 | 0.24566 (1) | 0.6124 m | 16.45° (2) | 0.037 | 4 m |
| E13 | 8 | 1e-4 | 42 | 120 / 76 (early stop) | 0.13842 (26) | 0.4056 m (66) | 9.18° (60) | 0.142 | 2 h 57 m |
| E14 | 8 | 1e-4 | 42 | 120 / 116 (early stop) | 0.13643 (66) | 0.4017 m (66) | 8.78° (105) | 0.153 | 4 h 29 m |

E14 usa además `weight_decay = 1e-4` (L2 dentro de Adam, no AdamW). Medianas destacadas: posición 35.08 cm (E08), angular 8.452° (E14).

### 4.2 Hallazgos por bloque

**Pruebas del pipeline (E01, E02, E12).** E01 validó la integración dataset–modelo–entrenamiento. E02 (batch 1) fue muy inestable (val_loss 3.09, 80° de error); una explicación plausible es Batch Normalization con lotes de 1, pero no se aisló la causa. E12 fue un smoke test de las funciones nuevas, no un modelo competitivo.

**Batch size y learning rate (E03–E07, E09).** Con 10 épocas batch 8 (E04) rindió mejor que batch 4 (E05) en val_loss (0.160 vs 0.171). Entre learning rates, 1e-4 fue mejor que 2e-4 y 5e-5 no mostró mejora consistente (E09 > E08 en val_loss). Nota: E06 y E07 usan batch 4 y E08 batch 8, por lo que la comparación de LR en la figura 2 no es estrictamente "una variable a la vez"; conviene aclararlo al presentarla.

**Más épocas (E10).** Las primeras 80 épocas son idénticas a E08 (diferencia máxima en val_loss 0.0), lo que confirma reproducibilidad con la misma semilla. De la época 81 a la 300 el val_loss final sube a 0.186 (+57.8 % sobre el mínimo) mientras train_loss baja a 0.0127. El menor error de posición (0.3724 m) se gana por ~5 mm sobre E08 y a un costo de 3.7× el tiempo.

**Semilla (E11).** Con seed 123 el mejor val_loss fue 0.1207 (ép. 164) y el mejor error angular 8.22°. La dispersión entre semillas es del mismo orden que las mejoras buscadas con hiperparámetros, de modo que diferencias de pocos milímetros o décimas de grado no son concluyentes con una sola semilla por configuración.

**Scheduler y early stopping (E13).** El LR cayó en las épocas 35, 46, 57 y 68 (1e-4 → 6.25e-6); el entrenamiento paró en la época 76 (patience 50; el mejor val_loss fue en la 26). No superó a E08.

**Weight decay (E14).** Caídas de LR en las épocas 28, 54, 78, 89, 100 y 111; paró en la 116. Mejoró la mediana angular, pero no superó a E08 y el sobreajuste persistió.

### 4.3 Diagnóstico de sobreajuste
Umbrales heurísticos (`src/analysis_nn/__init__.py`):

| Constante | Valor |
|---|---:|
| `OVERFIT_GAP_THRESHOLD` | 0.05 |
| `OVERFIT_RISE_PCT` | 5 % |
| `UNDERFIT_MIN_LOSS` | 0.2 |
| `UNDERFIT_DROP_PCT` | 10 % |
| `CONVERGED_SLOPE_EPS` | 0.001 |
| `CONVERGED_CV_EPS` | 0.02 |

Resultado: **10 de 11 runs analizados se marcan como overfit**; solo E06 (10 épocas, LR 5e-5) no. Ningún run se marca como underfit. Convergencia: sí en E10, E11, E13 y E14.

Lectura crítica: el criterio `gap > 0.05` es estricto cuando train_loss es ~0.01, así que también marca corridas cortas (E04, E05) donde el modelo todavía mejoraba. Los umbrales sirven para comparar runs, no como prueba de sobreajuste. Lo realmente informativo es la evolución de val_loss: se estanca cerca de 0.12 desde la época ~67 mientras train_loss sigue bajando.

### 4.4 Referencia actual
**E08** (batch 8, LR 1e-4, seed 42, 80 épocas), checkpoint de la época 67: val_loss 0.118176 y error medio de posición 0.3771 m. El error angular de 8.81° corresponde a la época 74; el de la época 67 puede ser distinto y debe medirse evaluando ese checkpoint.

---

### 4.5 Evolución por etapas de desarrollo
| Etapa | Runs | Qué se implementó o probó |
|---|---|---|
| 1. Modelo de regresión de pose | E01–E03 | ResNet-18 preentrenada, cabeza de 7 salidas, normalización de coordenadas, pérdida combinada, métricas de posición y orientación |
| 2. Hiperparámetros | E04–E09 | Batch 4 y 8, LR 5e-5 / 1e-4 / 2e-4, 10 y 80 épocas |
| 3. Comportamiento del modelo | E10–E11 | 300 épocas, curvas train/val, efecto de la semilla; se detecta que val empeora mientras train baja |
| 4. Automatización | E12–E13 | `ReduceLROnPlateau` y early stopping configurables por terminal |
| 5. Regularización | E14 | `weight_decay = 1e-4` (L2 dentro de Adam) |

Entre E13 y E14, E14 mejoró ligeramente las tres métricas (val_loss 0.1364 vs 0.1384; posición 0.4017 vs 0.4056 m; orientación 8.78° vs 9.18°), pero entrenó 116 épocas frente a 76. Ambos runs aparecen marcados con overfit en el análisis automático.

## 5. Análisis de por qué se estancó

Hipótesis a contrastar (no son conclusiones):

1. **Pocos datos y poca diversidad:** ≈ 869 imágenes de train de 4 trayectorias y validación en una sola secuencia. La red memoriza (train_loss ≈ 0.004–0.016).
2. **Sin aumento de datos.** Solo resize y normalización. Aumentos fotométricos (brillo/contraste, ya que hay videos de día y de noche) podrían ayudar; los geométricos (flip, rotaciones) alterarían la relación imagen–pose y no deben aplicarse sin ajustar las etiquetas.
3. **Techo de ruido en las etiquetas:** COLMAP + alineación con residuo ~2–3 cm, 8 poses sospechosas y filtrado imperfecto. Los errores de validación (~37 cm) son mucho mayores que el ruido de alineación, así que el techo de etiquetas no es el límite principal.
4. **Formulación de la pérdida:** pesos fijos 1:1 entre posición (unidades de std) y orientación. PoseNet2 propone pesos aprendibles; es el primer experimento a probar (ver 6.4) porque es barato y separa el efecto de la pérdida del del backbone.
5. **Arquitectura / regresión directa:** regresar pose desde una capa global (`fc` sobre ResNet-18) es la formulación original de PoseNet; sus evoluciones cambian backbone, pérdida o añaden restricciones geométricas.
6. **Validación estrecha y ruidosa:** una secuencia, una semilla por configuración → difícil distinguir mejoras reales.
7. **Preprocesado:** el resize a 224 × 224 deforma la relación de aspecto 856 × 480.

Los asesores sospechan de ResNet como causa; las hipótesis 1, 2, 4 y 6 son independientes de la arquitectura y se pueden controlar sin cambiarla.

---

## 6. Nueva línea: PoseNet y evoluciones (10–11 nov 2026)

> Resultado de una investigación preliminar. Las citas y los repositorios deben verificarse (autores, estado, versión de PyTorch) antes de basar decisiones en ellos.

### 6.1 Matiz conceptual
PoseNet (Kendall et al., ICCV 2015) ya es una CNN que regresa `[x, y, z, qx, qy, qz, qw]` con un backbone (GoogLeNet). El modelo actual (ResNet-18 + `fc` de 7 salidas) es, en esencia, un PoseNet con otro backbone. Las evoluciones cambian sobre todo la **función de pérdida**, la **arquitectura** y el uso de **restricciones geométricas**. Por tanto conviene probar la pérdida antes de culpar solo a ResNet.

### 6.2 Línea de evolución
| Método | Año | Aporte principal | Costo para este proyecto |
|---|---|---|---|
| PoseNet | 2015 | Regresión directa de pose con una CNN | Bajo (punto de partida) |
| Bayesian PoseNet | 2016 | Incertidumbre con MC dropout | Bajo–medio |
| PoseNet2 (pérdida geométrica) | 2017 | Pesos **aprendibles** para posición y rotación | **Muy bajo**: pocas líneas sobre el modelo actual |
| PoseLSTM | 2017 | LSTM sobre las características | Medio |
| MapNet | 2018 | Pérdida absoluta + restricciones de pose relativa entre imágenes | Medio; encaja con secuencias de video |
| AtLoc | 2020 | Atención sobre las características | Medio–alto |
| MS-Transformer | 2021 | Transformers, varias escenas a la vez | Alto |
| Direct-PoseNet / DFNet / NeFeS | 2021–2024 | Refinamiento con NeRF | Muy alto (GPU, PyTorch 1.11+/CUDA 11.3+) |

### 6.3 Implementaciones PyTorch encontradas (por verificar)
- PoseNet / PoseLSTM: repositorios como `youngguncho/PoseNet-Pytorch` y otro de PoseLSTM/PoseNet.
- MapNet: repositorio oficial (~359 estrellas, sin actualizaciones desde 2021).
- AtLoc y MS-Transformer: repositorios disponibles; hay versiones de PoseNet y MS-Transformer con un dataloader común.
- Direct-PoseNet, DFNet y NeFeS: requieren GPU y NeRF; **descartados por ahora** (el entrenamiento local es en CPU).

### 6.4 Plan de acción (orden propuesto)
1. **Experimento 1: pérdida con pesos aprendibles sobre ResNet-18.** Se reemplaza `L = L_pos + L_ori` por
   ```text
   L = L_pos · exp(−ŝx) + ŝx + L_ori · exp(−ŝq) + ŝq
   ```
   con `ŝx` y `ŝq` como parámetros entrenables (añadidos al optimizador, sin weight decay). Para que la comparación sea justa: mismo split, batch 8, LR 1e-4, 80 épocas, seed 42 contra E08, y repetir con otra semilla dado que hay variabilidad entre semillas. Las inicializaciones típicas de la literatura (p. ej. `ŝx = 0`, `ŝq = −3`) deben verificarse. Un run de 80 épocas en CPU toma ~3 h.
   - Si mejora claramente, el cuello de botella era la pérdida.
   - Si no mejora, apunta más al dato (cantidad/diversidad) o al backbone.
2. **Reproducir un PoseNet/PoseLSTM en PyTorch en Colab (T4)** con una escena pequeña de 7-Scenes (Chess), y verificar que los errores se acerquen a los del artículo antes de pasar a `v2_0`.
3. **MapNet** como siguiente paso (mismo formato de datos; su pérdida relativa puede aprovechar las secuencias de video, lo que exige conservar el orden de los frames).
4. **AtLoc o MS-Transformer** solo si 2 y 3 funcionan.
5. Adaptar la variante elegida a `v2_0` y compararla contra E08 (mismo split, etiquetas y métricas).

### 6.5 Precauciones al adaptar código ajeno
- **Orden del cuaternión:** muchos repositorios usan `[qw, qx, qy, qz]`; este proyecto usa `[qx, qy, qz, qw]`.
- **Convención de pose y unidades:** 7-Scenes usa su propio marco en metros; no mezclar con FLU.
- **Pérdida y q ≡ −q:** comprobar que la pérdida del código ajeno la respeta.
- **Normalización de posición:** documentar si se usa media/std de train.
- **Tiempo disponible:** quedan ~16 días hasta el 27 nov; priorizar los pasos 1 a 3 y dejar 4 y 5 como opcionales según resultados.

Criterios para elegir la variante final: una sola imagen de entrada, salida posición + cuaternión, código reproducible en PyTorch, entrenable en Colab o en un tiempo razonable en CPU.

**Estado:** el paso 1 es el siguiente a ejecutar; ninguna variante ha sido implementada todavía.

---

## 7. Checkpoints a evaluar antes de declarar un ganador

Evaluar todos sobre exactamente las mismas imágenes de validación y guardar predicciones por imagen:

| Checkpoint | Motivo |
|---|---|
| E08 / E10 ép. 67 | Mejor val_loss (idéntico) |
| E10 ép. 164 | Menor error de posición |
| E11 ép. 164 y 279 | Mejor val_loss de E11 y menor error angular |
| E14 ép. 66 y 105 | Mejor val_loss de E14 y mejor error angular |

**Nota:** `best_model.pth` guarda solo el mejor val_loss. Los checkpoints de las épocas 164, 279 o 105 no existen como archivos a menos que se hayan guardado aparte; en caso contrario se necesitarían reentrenamientos.

---

## 8. Registro cronológico

| Fecha | Avance técnico |
|---|---|
| 7 oct 2026 | Entorno `.venv`, PyTorch/torchvision, ResNet-18 verificado en laptop (CPU) y Colab (T4). |
| 14 oct 2026 | Primera ronda (`v1_0`): 12 videos, COLMAP conjunto (646 frames, 609 registrados), poses exportadas y split por video. Descartada después. |
| 21 oct 2026 | Baseline ResNet-18 de 20 épocas sobre la primera ronda (mejor época 16, val_loss 0.0915). Resultados de esa ronda no son comparables con v2_0. |
| 28 oct 2026 | Nueva captura `v2_0`: 5 videos (1,299 frames) + 15 referencias físicas; se rehízo el dataset porque `v1_0` no tenía distancias físicas y el ambiente había cambiado. |
| 28 oct – 4 nov | COLMAP sobre `v2_0` (1,189 registradas), alineación Umeyama a FLU, validación de poses. |
| 4 – 9 nov | Experimentos E01–E14 con ResNet-18 sobre `v2_0`. |
| 10 nov 2026 | Gráficas y análisis de los 14 entrenamientos; se detecta estancamiento y sobreajuste. Reunión con asesores: pasar a PoseNet y sus evoluciones. |
| 11 nov 2026 | Documentación: README, Progress Report y Technical Advances. Investigación preliminar de PoseNet y evoluciones incorporada (sección 6); primer experimento propuesto: pérdida con pesos aprendibles sobre ResNet-18. |

---

## 9. Pendientes técnicos

- [ ] Verificar split en `poses_split.csv` (869 vs 884).
- [ ] Inspeccionar las 8 poses sospechosas.
- [ ] Validar la alineación con referencias no usadas en el ajuste.
- [ ] Evaluar checkpoints candidatos de ResNet-18 con un script común.
- [ ] Probar aumento fotométrico y comparar (si se sigue con ResNet-18 como control).
- [ ] Experimento 1: pérdida con pesos aprendibles (PoseNet2) sobre ResNet-18 vs E08.
- [ ] Reproducir PoseNet/PoseLSTM en Colab con 7-Scenes (Chess) y verificar contra el artículo.
- [ ] Probar MapNet; AtLoc/MS-Transformer solo si hay tiempo.
- [ ] Adaptar la variante elegida a `v2_0` y comparar con E08.
- [ ] Definir y evaluar el test independiente (~800 imágenes mencionadas, sin manifiesto documentado).
- [ ] Medir latencia de inferencia y tamaño del modelo.
- [ ] Definir la transformación cámara→cuerpo del Bebop.
- [ ] Probar inferencia offline y, si el tiempo lo permite, integración ROS Noetic.

---

## 10. Correcciones a versiones previas de la documentación

- La figura 2 (panel B, barrido de LR) compara E06/E07 (batch 4, 10 épocas) con E08 (batch 8, 80 épocas), así que no aísla el efecto del learning rate.
- El docstring de `compare_runs.py` indica `src.neuralNetwork.analysis.compare_runs`, pero el módulo importa de `src.analysis_nn`. Comando correcto: `python -m src.analysis_nn.compare_runs outputs_nn/resnet18`.
- Los errores "mejores" por run no deben presentarse como un único modelo.
