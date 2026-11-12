# Progress Report — Estimación visual de pose 6-DoF de un dron mediante aprendizaje profundo

| Campo | Detalle |
|---|---|
| **Practicante** | Ricardo Ramos Morales |
| **Asesor** | Dr. José Martínez Carranza |
| **Institución** | INAOE |
| **Periodo de prácticas** | 1 de septiembre – 27 de noviembre de 2026 |
| **Corte de este reporte** | 11 de noviembre de 2026 |

> Este documento resume **actividades y logros** por etapa. El detalle de cada experimento está en `Technical_Advances.md`.

---

## 1. Resumen ejecutivo

El proyecto busca estimar la pose 6-DoF (posición + cuaternión) de la cámara de un Parrot Bebop 2 a partir de una sola imagen, sin odometría ni sensores externos en la inferencia.

A la fecha de corte:

- Se construyó el dataset `v2_0` (cinco recorridos y 15 fotos de referencia con posición física medida) y se reconstruyó con COLMAP (1,189 imágenes registradas).
- Se alinearon las poses al marco físico FLU en metros (Umeyama/SVD), generando las etiquetas de entrenamiento.
- Se implementó un pipeline completo de entrenamiento en PyTorch y se ejecutaron **14 experimentos con ResNet-18**.
- El mejor baseline (E08) alcanza val_loss 0.1182, ≈ 37.7 cm de error medio de posición y ≈ 8.8° de error angular en una secuencia de validación.
- Al graficar los 14 entrenamientos (10 nov) se observó que, pese a variar varios hiperparámetros, **ya no había mejora consistente** y varios runs mostraban sobreajuste.
- En la reunión del 10 nov con los asesores se decidió **dejar de ajustar ResNet-18 e investigar PoseNet y sus evoluciones**.

---

## 2. Avance frente al cronograma

| Periodo | Actividad planeada | Estado | Comentario |
|---|---|---|---|
| 1 – 25 sep | Estado del arte y herramientas | Completado | |
| 21 sep – 13 oct | Captura, extracción de frames, COLMAP | Completado | Primera ronda `v1_0`: 12 videos (646 frames, 609 registrados); descartada por falta de referencias físicas |
| 7 – 20 oct | Entorno, pruebas de entrenamiento, pipeline PyTorch | Completado | Verificado en CPU y Colab (T4) |
| 20 oct – 4 nov | Nueva captura `v2_0` + referencias físicas + alineación métrica | Completado | Ver sección 3; se rehízo el dataset porque `v1_0` no tenía distancias físicas para la escala y el ambiente había cambiado |
| 4 – 10 nov | Entrenamientos ResNet-18 y análisis | Completado | 14 runs, herramientas de análisis |
| 10 – 13 nov | Cerrar etapa ResNet-18, investigar alternativas | En curso | Investigación de PoseNet iniciada |
| 9 – 20 nov | Inferencia e integración ROS/Bebop | Pendiente | Depende de la arquitectura final |
| 16 – 23 nov | Validación final, métricas, visualizaciones | Pendiente | |
| 23 – 27 nov | Informe final, README, limpieza | Pendiente | |

---

## 3. Actividades realizadas y logros

### 3.1 Dataset y reconstrucción 3D
**Actividades**
- Captura de recorridos con el Bebop 2 y extracción de frames a ~2 FPS con recortes por video.
- Registro de 15 fotografías con posiciones físicas conocidas (origen en la esquina de las redes).
- Reconstrucción conjunta con COLMAP y filtrado de la reconstrucción.

**Logros**
- 1,314 imágenes en total; **1,189 registradas (≈ 90.5 %)**; 107,703 puntos 3D; error medio de reproyección 0.966 px.
- Split por secuencia completa para evitar fuga entre frames consecutivos: `bebop_001_val` (305 registradas) para validación; `bebop_002`–`005_train` (138, 217, 238, 276) para entrenamiento.

### 3.2 Alineación COLMAP → marco físico FLU
**Actividades**
- Implementación de `align_colmap_to_flu.py` con el algoritmo de Umeyama (SVD) para estimar escala, rotación y traslación.
- Validación de las poses resultantes.

**Logros**
- Escala **0.394834 m/unidad COLMAP**; MAE 3D 1.97 cm; RMSE 3D 2.68 cm.
- Etiquetas en metros en `poses_drone_FLU.csv`: 1,181 poses válidas, 8 sospechosas, 0 inválidas.

**Aclaración:** estos errores se calculan con las mismas 15 referencias usadas en el ajuste; no son una validación independiente.

### 3.3 Pipeline de aprendizaje profundo
**Actividades**
- Dataset PyTorch (resize 224 × 224, normalización ImageNet), normalización de posición con estadísticas solo de train.
- Modelo `ResNetPoseRegressor` (ResNet-18 preentrenada, 7 salidas) y pérdida compuesta MSE de posición + `1 − |q·q̂|`.
- Entrenamiento con Adam, y después ReduceLROnPlateau, early stopping y weight decay.
- Exportación de predicciones por imagen en validación.

**Logros**
- Pipeline reproducible con carpeta, historial y `run_info.json` por ejecución.

### 3.4 Experimentación con ResNet-18 (E01–E14)
**Actividades:** exploración de batch size, learning rate, número de épocas, semilla, scheduler, early stopping y weight decay (entrenamientos en CPU; de 4 min a 11 h 42 min cada uno).

**Logros / hallazgos**
- Batch 8 y LR 1e-4 resultaron la mejor combinación explorada; batch 1 fue inestable (E02) y LR 5e-5 / 2e-4 no mejoraron.
- **E08** (80 épocas) es el checkpoint de referencia: mejor val_loss 0.118176 (época 67).
- E10 (300 épocas) repite el mismo mejor val_loss: las primeras 80 épocas de E08 y E10 son idénticas (comprobado), y entrenar más solo aumentó el sobreajuste.
- Cambiar la semilla (E11) modificó los resultados: hay variabilidad entre semillas.
- Scheduler, early stopping y weight decay (E13, E14) funcionaron técnicamente pero no superaron a E08.

| Indicador | Resultado | Run |
|---|---:|---|
| Menor val_loss | 0.118176 | E08 y E10 |
| Menor error medio de posición | 0.3724 m | E10 |
| Menor error medio angular | 8.2238° | E11 |
| Menor mediana de posición | 35.08 cm | E08 |
| Menor mediana angular | 8.452° | E14 |

Los mínimos individuales provienen de épocas distintas y no deben combinarse como si fueran un solo modelo.

### 3.5 Herramientas de análisis
- `analyze_run.py`: 5 gráficas + dashboard + `report.md` por entrenamiento y `all_reports.md` consolidado.
- `compare_runs.py`: comparaciones entre runs, ranking por métricas, diagnóstico de sobreajuste, tiempos de entrenamiento y sanity checks (E08 vs E10, scheduler de E13/E14).
- Con ellas se pasó de revisar corridas sueltas a un historial comparable (hecho el 10 nov).

---

## 4. Problemas encontrados y cómo se abordaron

| Problema | Acción / estado |
|---|---|
| Escala y marco arbitrarios de COLMAP | Alineación con 15 referencias físicas (Umeyama). |
| Entrenar en CPU (hasta 11.7 h por run) | Se priorizaron configuraciones; se evitaron más corridas largas. |
| Sobreajuste (train_loss ≈ 0.004–0.016 vs val_loss ≈ 0.12–0.19) | Probados scheduler, early stopping y weight decay; sin mejora concluyente. |
| Mejora estancada tras 14 runs | Reunión con asesores y cambio de línea a PoseNet y evoluciones. |
| Discrepancia de 15 imágenes en train de `v2_0` (869 por suma de secuencias vs 884 en un resumen previo) | Pendiente de verificar los conteos en `poses_split.csv`. |
| Ocho poses sospechosas (p. ej. salto atípico en `bebop_001_val_frame_00043`) | Pendiente de inspección. |

---

## 5. Limitaciones actuales

- Validación con **una sola secuencia**; no hay test independiente evaluado (se mencionan ~800 imágenes reservadas, pero su manifiesto y poses aún no están definidos aquí).
- Las etiquetas son estimaciones de COLMAP, no ground truth independiente.
- Las poses describen la cámara; falta la transformación cámara→cuerpo del dron.
- Sin integración ROS todavía.

---

## 6. Plan de las siguientes semanas

**Semana del 11 al 15 nov**
- Experimento 1: pérdida con pesos aprendibles (PoseNet2) sobre ResNet-18, comparada contra E08 (mismo split y configuración). Es un cambio de pocas líneas y indica si el problema es el backbone o la pérdida.
- Leer el artículo original de PoseNet y los trabajos que lo citan (Bayesian PoseNet, PoseNet2, PoseLSTM, MapNet, AtLoc, MS-Transformer).
- Reproducir un PoseNet/PoseLSTM en PyTorch en Colab (T4) con una escena de 7-Scenes (Chess) y comprobar que se acerca a lo reportado en el artículo.
- Evaluar con un mismo script los checkpoints candidatos de ResNet-18 (E08/E10 ép. 67, E10 ép. 164, E11 ép. 164 y 279, E14 ép. 66 y 105) y analizar errores por imagen.

**Semana del 16 al 22 nov**
- Probar MapNet (siguiente paso lógico); AtLoc o MS-Transformer solo si lo anterior funciona. Direct-PoseNet, DFNet y NeFeS se descartan por ahora (NeRF y GPU).
- Adaptar la variante elegida a `v2_0` y compararla con E08 (mismo split y métricas).
- Revisar poses sospechosas y definir el test independiente.
- Medir latencia y tamaño del modelo; inferencia offline.

**23 al 27 nov**
- Evaluación final, gráficas, informe final y limpieza del repositorio.
- ROS Noetic solo si hay un modelo validado y tiempo.

---

## 7. Evidencias

| Evidencia | Ubicación |
|---|---|
| Reportes por entrenamiento | `analysis/runs/<run>/report.md` |
| Reporte consolidado | `analysis/all_reports.md` |
| Comparación entre runs | `outputs_nn/resnet18/analysis_compare/` |
| Poses en metros | `results_colmap/v2_0/alignment/poses_drone_FLU.csv` |
| Validación de poses | `results_colmap/v2_0/alignment/pose_validation_all.csv` |
| Checkpoints e historiales | `outputs_nn/resnet18/<run>/` |
