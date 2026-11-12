# Drone 6-DoF Pose Estimation

Estimación monocular de la pose 6-DoF de la cámara de un dron Parrot Bebop 2 a partir de una sola imagen, usando aprendizaje profundo (PyTorch) y etiquetas obtenidas con reconstrucciones Structure-from-Motion de COLMAP.

| Información | Detalle |
|---|---|
| **Institución** | Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE) |
| **Periodo** | 1 de septiembre – 27 de noviembre de 2026 |
| **Autor** | Ricardo Ramos Morales |
| **Asesor** | Dr. José Martínez Carranza |
| **Plataforma** | Parrot Bebop 2 |
| **Última actualización** | 11 de noviembre de 2026 |

> Este README explica **qué es el proyecto y cómo usar el repositorio**. El avance por periodo está en `Progress_Report.md` y el historial técnico detallado de experimentos en `Technical_Advances.md`.

---

## 1. ¿Qué hace el proyecto?

Dada una imagen RGB de la cámara frontal del Bebop 2 en un entorno interior, el modelo regresa:

```text
[x, y, z, qx, qy, qz, qw]
```

- **Posición** `x, y, z` en metros, en el marco físico **FLU** (Forward–Left–Up) con origen en la esquina del espacio donde se juntan las redes.
- **Orientación** como cuaternión unitario (`q` y `-q` representan la misma rotación; la pérdida y las métricas lo respetan).

Durante la inferencia final el modelo **solo recibe la imagen**: no usa odometría, GPS ni sensores externos. Esos recursos solo sirven en desarrollo para comparar y validar.

> **Alcance:** la pose estimada es la de la **cámara**. Todavía no se aplica la transformación cámara→cuerpo del dron, así que no debe afirmarse que la salida es la pose calibrada del centro del Bebop.

---

## 2. Pipeline

```text
Videos Bebop 2 (5 recorridos) + 15 fotos de referencia con posición medida
        │
        ▼
Extracción de frames (~2 FPS, recortes por video)
        │
        ▼
COLMAP (características → matching → SfM)  →  poses de cámara (escala arbitraria)
        │
        ▼
Alineación de similitud 3D (Umeyama/SVD) con las 15 referencias  →  metros, marco FLU
        │
        ▼
Validación de poses (válidas / sospechosas / inválidas)
        │
        ▼
Split por secuencia completa (train / val)  →  poses_split.csv
        │
        ▼
Dataset + DataLoader PyTorch  →  Regresor de pose  →  Evaluación
```

Las etiquetas vienen de COLMAP, por lo que **no son ground truth independiente**: la red aprende también los errores de la reconstrucción.

> **Historial de datasets:** una primera ronda (`v1_0`, 12 videos) se descartó porque no había distancias físicas para fijar la escala en COLMAP y el ambiente donde se capturó cambió. Se rehízo como `v2_0` (5 videos + 15 referencias medidas). Todos los resultados de este repositorio usan `v2_0`.

---

## 3. Estado actual (11 nov 2026)

| Componente | Estado |
|---|---|
| Dataset `v2_0` (captura, frames, referencias) | Completado |
| Reconstrucción COLMAP | Completada: 1,189 / 1,314 imágenes registradas (~90.5 %) |
| Alineación COLMAP → FLU | Implementada (escala 0.394834 m/unidad, RMSE 3D 2.68 cm sobre las propias referencias) |
| Pipeline de entrenamiento ResNet-18 | Implementado, 14 experimentos (E01–E14) |
| Herramientas de análisis de entrenamientos | Implementadas |
| Mejor baseline | **E08** (val_loss 0.1182, error medio de posición ≈ 37.7 cm, angular ≈ 8.8°) |
| Arquitectura siguiente (PoseNet o evolución) | En investigación |
| Evaluación en test independiente | Pendiente |
| Integración ROS Noetic | Pendiente |

**Resumen:** tras 14 entrenamientos con ResNet-18 (batch, learning rate, semilla, épocas, scheduler, early stopping y weight decay) la validación dejó de mejorar y se observó sobreajuste. El 10 de noviembre, con los asesores, se decidió detener los ajustes sobre ResNet-18 e investigar PoseNet y sus evoluciones.

**Matiz importante:** PoseNet es una CNN que regresa `[x, y, z, qx, qy, qz, qw]` con un backbone (GoogLeNet), así que el modelo actual ya es, en esencia, un PoseNet con backbone ResNet-18. Las evoluciones cambian sobre todo la función de pérdida, la arquitectura y las restricciones geométricas. Por eso el primer experimento será probar la pérdida con pesos aprendibles (PoseNet2) sobre el mismo ResNet-18, para saber si el backbone es realmente el problema.

---

## 4. Estructura del repositorio

```text
drone-6dof-pose-estimation/
├── dataset/                      # fuera de Git
│   └── v2_0/
│       ├── frames_all/           # imágenes planas usadas por el Dataset
│       ├── poses_split.csv       # poses FLU + columna split
│       └── colmap_reference_positions.txt
├── results_colmap/               # fuera de Git
│   └── v2_0/alignment/
│       ├── poses_drone_FLU.csv
│       ├── pose_validation_all.csv
│       └── similarity_transform_colmap_to_FLU.txt
├── outputs_nn/                   # fuera de Git (checkpoints e historiales)
│   └── resnet18/<run>/{best_model.pth, history.json, run_info.json, val_predictions.csv}
├── analysis/                     # reportes y figuras generadas
├── src/
│   ├── pre_colmap/extract_frames.py
│   ├── post_colmap/
│   │   ├── filter/filter_colmap_reconstruction.py
│   │   ├── align_colmap_to_flu.py
│   │   ├── split_dataset.py
│   │   └── validate_poses.py
│   ├── neuralNetwork/{dataset.py, model.py, train.py, train_2.py, test_resnet18.py}
│   └── analysis_nn/{__init__.py, analyze_run.py, compare_runs.py}
├── requirements.txt
└── README.md
```

`dataset/`, `outputs_nn/` y `results_colmap/` ocupan mucho espacio y se mantienen fuera de Git. Verifica el contenido real de tu copia antes de asumir que todas las rutas existen.

---

## 5. Entorno

| Componente | Especificación |
|---|---|
| Equipo | HP Pavilion 15-cw1xxx, AMD Ryzen 5 3500U, 24 GB RAM |
| GPU | Radeon Vega 8 integrada (sin CUDA); entrenamiento en **CPU** |
| SO / Python | Ubuntu 24.04 / 3.12.3 |
| PyTorch | 2.14.1+cu130 (reportado en el entorno local) |
| COLMAP | 3.9.1 (sin CUDA) |

```bash
cd ~/drone-6dof-pose-estimation
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 6. Uso

### 6.1 Extraer frames
```bash
python src/pre_colmap/extract_frames.py --fps 2 --cut-mode per-video
```
Usa `--clean-output` solo si quieres borrar los `.jpg` existentes.

### 6.2 Alinear poses COLMAP al marco FLU
```bash
python src/post_colmap/align_colmap_to_flu.py
```
Genera `results_colmap/v2_0/alignment/poses_drone_FLU.csv`.

### 6.3 Entrenar
```bash
# Prueba corta
python src/neuralNetwork/train_2.py --epochs 2 --seed 42 --run-tag smoke_test

# Configuración de referencia (E08)
python src/neuralNetwork/train_2.py --epochs 80 --batch-size 8 --lr 1e-4 --seed 42 \
    --early-stopping-patience 0

# Scheduler + early stopping + weight decay (E14)
python src/neuralNetwork/train_2.py --epochs 120 --batch-size 8 --lr 1e-4 --seed 42 \
    --weight-decay 1e-4 --early-stopping-patience 50 --run-tag e14_wd
```
Cada ejecución crea `outputs_nn/resnet18/<version>_run_NN_.../`. Confirma los argumentos con `--help`. Nota: `train_2.py` siempre crea un scheduler `ReduceLROnPlateau`; E01–E11 no tenían scheduler (su history no registra LR), así que antes de reproducir E08 con este script conviene verificar con qué script y configuración se entrenó realmente.

### 6.4 Analizar entrenamientos
```bash
python -m src.analysis_nn.analyze_run --all outputs_nn/resnet18
python -m src.analysis_nn.compare_runs outputs_nn/resnet18
```
Producen curvas de pérdida, errores de posición y orientación, learning rate, gap de generalización, dashboards por run, `all_reports.md` y figuras comparativas.

---

## 7. Representación y métricas

- **Pérdida:** `L = MSE(posición normalizada) + (1 − |q_pred · q_real|)`, con pesos 1:1. La posición se normaliza con media/desviación calculadas **solo con train**.
- **Error de posición:** distancia euclidiana en metros.
- **Error angular:** `2·arccos(|q_pred · q_real|)` en grados.
- **Entrada:** 224 × 224 con normalización ImageNet; ResNet-18 preentrenada con capa final de 7 salidas.
- **Optimizador:** Adam (no AdamW). El `weight_decay` de E14 es L2 dentro de Adam.

---

## 8. Limitaciones conocidas

1. La validación es **una sola secuencia** (`bebop_001_val`); no demuestra generalización a otras rutas, iluminación o entornos.
2. El conjunto de **test independiente** aún no está definido ni evaluado.
3. Los residuos de alineación (RMSE 2.68 cm) usan las mismas 15 referencias del ajuste; **no son validación independiente**.
4. Las etiquetas son estimaciones de COLMAP; ocho poses están marcadas como sospechosas y falta revisarlas.
5. Los mejores errores de posición y orientación por run provienen de épocas distintas; no describen un único checkpoint.
6. Falta la transformación cámara→cuerpo del Bebop.
7. Hay una discrepancia de 15 imágenes entre la suma de train por secuencia de `v2_0` (869) y un total reportado en un resumen previo (884); verificar los conteos reales en `poses_split.csv`.

---

## 9. Próximos pasos

- [ ] **Experimento 1 (barato):** ResNet-18 con pérdida de pesos aprendibles (PoseNet2, Kendall & Cipolla 2017) y comparar contra E08 con el mismo split y configuración.
- [ ] Reproducir un PoseNet/PoseLSTM en PyTorch en Colab (T4) sobre una escena de 7-Scenes (p. ej. Chess) y verificar que el error se acerque al del artículo.
- [ ] Siguiente candidato: MapNet (pérdida relativa entre imágenes de una secuencia); AtLoc o MS-Transformer solo si lo anterior funciona. Descartar por ahora Direct-PoseNet/DFNet/NeFeS (NeRF, requieren GPU).
- [ ] Adaptar la variante elegida a `v2_0`.
- [ ] Comparar contra E08 con el mismo split, etiquetas y métricas.
- [ ] Evaluar los checkpoints candidatos de ResNet-18 sobre exactamente las mismas imágenes de validación.
- [ ] Revisar las 8 poses sospechosas y confirmar el split.
- [ ] Definir y evaluar un conjunto de test independiente.
- [ ] Medir latencia de inferencia y tamaño del modelo.
- [ ] Probar inferencia offline e integración ROS Noetic si el tiempo lo permite.

---

## 10. Documentación relacionada

| Documento | Propósito |
|---|---|
| `README.md` | Explicar el proyecto y el funcionamiento del repositorio. |
| `Progress_Report.md` | Actividades y logros para evaluar el avance. |
| `Technical_Advances.md` | Historial técnico detallado de experimentos y desarrollos. |
| `Final_Report.docx` | Informe formal final conforme a los 11 apartados institucionales. |
