# Informe de avances técnicos
## Estimación de pose 6-DoF de un dron mediante visión monocular y aprendizaje profundo

**Proyecto:** `drone-6dof-pose-estimation`  
**Institución:** Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE)  
**Periodo de prácticas:** 1 de septiembre – 27 de noviembre de 2026  
**Responsable del proyecto:** Ricardo Ramos Morales  
**Asesor:** Dr. José Martínez Carranza  
**Última actualización:** 11 de noviembre de 2026

---

# 1. Objetivo general del proyecto

El objetivo del proyecto es desarrollar un sistema capaz de estimar la pose tridimensional de un dron Parrot Bebop 2 utilizando una sola imagen de su cámara frontal.

El sistema debe estimar seis grados de libertad (6-DoF):

- **Posición:** coordenadas X, Y y Z.
- **Orientación:** rotación respecto a los tres ejes, representada mediante un cuaternión.

La salida del modelo tiene la siguiente estructura:

`[x, y, z, qx, qy, qz, qw]`

La estimación debe expresarse en metros y utilizar un sistema de coordenadas físico consistente.

Una condición importante del proyecto es que, durante la inferencia final, el sistema no dependa de odometría, GPS ni sensores externos de posicionamiento. La información necesaria para estimar la pose debe obtenerse directamente de la imagen.

La odometría y las mediciones externas pueden utilizarse durante el desarrollo para realizar comparaciones y validar los resultados, pero no forman parte de la entrada del modelo final.

# 2. Preparación del dataset y reconstrucción 3D

## 2.1 Captura de imágenes

Se utilizó un dron Parrot Bebop 2 equipado con una cámara frontal fija.

Se realizaron diferentes recorridos dentro de un entorno interior, obteniendo un conjunto de imágenes que posteriormente se utilizó para construir una reconstrucción tridimensional mediante COLMAP.

Para la versión `v2_0` se recopilaron:

- 1,314 imágenes originales.
- 15 imágenes de referencia con posiciones físicas conocidas.
- Aproximadamente 800 imágenes adicionales reservadas para la evaluación final.

## 2.2 Reconstrucción mediante COLMAP

Se utilizó COLMAP para realizar la reconstrucción Structure from Motion (SfM), permitiendo obtener:

- Posiciones estimadas de las cámaras.
- Orientaciones de las cámaras.
- Puntos tridimensionales del entorno.
- Relaciones geométricas entre las imágenes.

La reconstrucción permitió registrar 1,189 de las 1,314 imágenes originales, equivalente aproximadamente al 90.5 % del conjunto.

La reconstrucción utilizada corresponde a `dataset/v2_0`.

## 2.3 Separación de datos

Se estableció una separación de los datos por secuencia de video, evitando distribuir aleatoriamente imágenes de una misma trayectoria entre entrenamiento y validación.

| Grupo | Imágenes | Uso |
|---|---:|---|
| `bebop_001_val` | 305 | Validación |
| `bebop_002_train` | 138 | Entrenamiento |
| `bebop_003_train` | 217 | Entrenamiento |
| `bebop_004_train` | 238 | Entrenamiento |
| `bebop_005_train` | 276 | Entrenamiento |
| Referencias | 15 | Alineación física |

**Nota de revisión:** las cantidades individuales de los cuatro grupos de entrenamiento suman 869 imágenes, mientras que el resumen general del pipeline registra 884 imágenes de entrenamiento. Esta diferencia de 15 imágenes debe verificarse contra `poses_split.csv` antes de cerrar la versión definitiva del informe.

La separación por videos constituye una decisión metodológica importante, ya que reduce el riesgo de obtener resultados artificialmente buenos debido a imágenes prácticamente idénticas presentes tanto en entrenamiento como en validación.

El conjunto de prueba permanece reservado y no se ha utilizado para seleccionar modelos.

**Limitación:** la validación representa una única trayectoria de vuelo. Por lo tanto, todavía no se puede afirmar que el modelo generalice correctamente a otras trayectorias, condiciones de iluminación o posiciones del dron.

# 3. Conversión de poses COLMAP al sistema físico FLU

## 3.1 Problema inicial

COLMAP proporciona las poses de las cámaras dentro de su propio sistema de coordenadas, cuya escala y orientación no necesariamente corresponden a las dimensiones físicas del entorno.

Por esta razón, no era posible utilizar directamente sus coordenadas como etiquetas de entrenamiento expresadas en metros.

Fue necesario establecer una transformación entre:

1. El sistema de coordenadas de COLMAP.
2. El sistema físico de referencia del entorno.

## 3.2 Referencias físicas

Se utilizaron 15 posiciones de referencia medidas físicamente y distribuidas dentro del entorno.

Estas referencias permitieron establecer correspondencias entre posiciones conocidas en el mundo físico y las posiciones correspondientes dentro de la reconstrucción de COLMAP.

## 3.3 Algoritmo de alineación

Se implementó el algoritmo de alineación de Umeyama, utilizando descomposición en valores singulares (SVD).

El algoritmo permitió estimar una transformación de similitud 3D que contempla:

- Escala.
- Rotación.
- Traslación.

La transformación se utilizó para llevar las posiciones obtenidas por COLMAP al sistema físico FLU.

La convención física utilizada es:

- X: Forward (adelante).
- Y: Left (izquierda).
- Z: Up (arriba).

También se consideró la transformación entre la convención de coordenadas de cámara de COLMAP y el sistema físico del dron.

## 3.4 Resultados de alineación

| Métrica | Resultado |
|---|---:|
| Escala | 0.394834 m/unidad COLMAP |
| MAE 3D | 0.0197 m |
| RMSE 3D | 0.0268 m |
| Error relativo | 1.64 % |

Los resultados muestran una buena correspondencia entre las referencias físicas y la reconstrucción.

Sin embargo, es importante aclarar que estas métricas corresponden a los residuos del ajuste utilizando las mismas 15 referencias para estimar y evaluar la transformación. Por tanto, no constituyen una validación independiente.

## 3.5 Resultado obtenido

Se desarrolló el script:

`src/post_colmap/align_colmap_to_flu.py`

El resultado principal es:

`results_colmap/v2_0/alignment/poses_drone_FLU.csv`

Este archivo contiene las poses transformadas al sistema físico y constituye la fuente de etiquetas para entrenar la red neuronal.

**Conclusión de esta etapa:** se logró establecer una relación métrica entre la reconstrucción visual de COLMAP y el entorno físico, permitiendo utilizar las poses como etiquetas supervisadas.

# 4. Desarrollo del pipeline de aprendizaje profundo

Una vez obtenidas las etiquetas, se desarrolló un pipeline de entrenamiento para realizar regresión de pose a partir de imágenes RGB.

## 4.1 Arquitectura inicial

La primera arquitectura seleccionada fue ResNet-18, implementada mediante PyTorch.

Se desarrolló el modelo `ResNetPoseRegressor`, que utiliza una ResNet-18 como extractor de características y una capa de regresión para producir siete valores.

La salida corresponde a:

- Tres valores para posición.
- Cuatro valores para orientación mediante cuaterniones.

## 4.2 Preparación de imágenes

Las imágenes se procesan antes de ingresar al modelo:

- Redimensionamiento a 224 × 224 píxeles.
- Normalización utilizando los parámetros de ImageNet.
- Conversión a tensores de PyTorch.

## 4.3 Normalización de etiquetas

Las coordenadas de posición se normalizan utilizando la media y desviación estándar calculadas sobre el conjunto de entrenamiento.

Para `v2_0`:

| Coordenada | Media | Desviación estándar |
|---|---:|---:|
| X | 1.47235 | 1.05284 |
| Y | -1.17480 | 0.93395 |
| Z | 0.85308 | 0.57002 |

Los cuaterniones se normalizan para mantener su representación válida como orientación.

## 4.4 Función de pérdida

Se implementó una función de pérdida compuesta por dos componentes:

**Pérdida de posición:**

Error cuadrático medio entre las coordenadas predichas y las coordenadas reales normalizadas.

**Pérdida de orientación:**

Se utiliza una función basada en el valor absoluto del producto punto entre los cuaterniones normalizados:

`1 - |q_pred · q_real|`

El valor absoluto permite considerar la equivalencia entre los cuaterniones q y -q, que representan la misma orientación.

La pérdida total combina ambos componentes:

`L_total = L_position + L_orientation`

## 4.5 Optimizador y entrenamiento

Se utilizó:

`torch.optim.Adam`

No se utilizó AdamW.

La configuración base más competitiva fue:

| Parámetro | Valor |
|---|---|
| Optimizador | Adam |
| Learning rate | 0.0001 |
| Batch size | 8 |
| Seed | 42 |
| Weight decay | 0 |
| Arquitectura | ResNet-18 |
| Dataset | v2_0 |

Posteriormente se incorporaron mecanismos adicionales de control:

- ReduceLROnPlateau.
- Early Stopping.
- Weight Decay.

En el experimento E14 se utilizó `weight_decay = 1e-4` dentro de Adam. Esto debe describirse como regularización mediante weight decay en Adam, no como weight decay desacoplado de AdamW.

## 4.6 Hardware utilizado

Los entrenamientos documentados se realizaron localmente en CPU.

| Componente | Especificación |
|---|---|
| Procesador | AMD Ryzen 5 3500U |
| Memoria RAM | 24 GB |
| GPU | Radeon Vega 8 integrada |
| Sistema operativo | Ubuntu 24.04 |
| PyTorch | 2.14.1+cu130 |
| Dispositivo de entrenamiento | CPU |

Esta limitación de hardware tuvo un impacto considerable en los tiempos de entrenamiento, especialmente en los experimentos de 300 épocas.

# 5. Experimentación con ResNet-18

Se realizaron 14 ejecuciones experimentales para estudiar el comportamiento de la arquitectura y encontrar una configuración que permitiera obtener estimaciones de posición y orientación suficientemente precisas.

Los experimentos se organizaron progresivamente:

1. Comprobar el funcionamiento del pipeline.
2. Explorar diferentes tamaños de batch.
3. Comparar tasas de aprendizaje.
4. Incrementar el número de épocas.
5. Analizar la variabilidad entre semillas.
6. Incorporar mecanismos de regularización y control del entrenamiento.

## 5.1 Historial de experimentos

| Run | Batch | LR | Seed | Épocas | Modificación principal |
|---|---:|---:|---:|---:|---|
| E01 | 2 | 1e-4 | 42 | 2 | Primera prueba del pipeline |
| E02 | 1 | 1e-4 | 42 | 4 | Exploración de batch pequeño |
| E03 | 4 | 1e-4 | 42 | 80 | Entrenamiento prolongado |
| E04 | 8 | 1e-4 | 42 | 10 | Exploración de batch 8 |
| E05 | 4 | 1e-4 | 42 | 10 | Referencia de configuración |
| E06 | 4 | 5e-5 | 42 | 10 | Reducción del learning rate |
| E07 | 4 | 2e-4 | 42 | 10 | Incremento del learning rate |
| E08 | 8 | 1e-4 | 42 | 80 | Configuración base competitiva |
| E09 | 8 | 5e-5 | 42 | 80 | Comparación de learning rate |
| E10 | 8 | 1e-4 | 42 | 300 | Entrenamiento prolongado |
| E11 | 8 | 1e-4 | 123 | 300 | Comparación entre semillas |
| E12 | 8 | 1e-4 | 42 | 2 | Smoke test de nuevas funciones |
| E13 | 8 | 1e-4 | 42 | 120 | Scheduler y Early Stopping |
| E14 | 8 | 1e-4 | 42 | 120 | Scheduler, Early Stopping y WD |

Todos los experimentos utilizaron Adam.

## 5.2 Comparación de resultados

Las siguientes métricas corresponden a validación. Se muestra el mejor valor registrado para cada indicador, que no necesariamente corresponde a la misma época.

| Run | Mejor val_loss | Error medio posición | Error medio angular | Tiempo |
|---|---:|---:|---:|---:|
| E01 | 0.43314 | 0.8442 m | 24.99° | 5 min 52 s |
| E02 | 3.08903 | 2.2641 m | 80.27° | 17 min 08 s |
| E03 | 0.13575 | 0.3858 m | 9.66° | 3 h 22 min |
| E04 | 0.16001 | 0.4861 m | 10.88° | 23 min 18 s |
| E05 | 0.17057 | 0.4748 m | 12.45° | 24 min 59 s |
| E06 | 0.15670 | 0.4532 m | 14.91° | 25 min 49 s |
| E07 | 0.20869 | 0.5431 m | 12.52° | 25 min 28 s |
| E08 | **0.11818** | **0.3771 m** | 8.81° | 3 h 07 min |
| E09 | 0.12999 | 0.4230 m | 9.33° | 3 h 08 min |
| E10 | **0.11818** | **0.3724 m** | 8.81° | 11 h 42 min |
| E11 | 0.12067 | 0.3857 m | **8.22°** | 11 h 40 min |
| E12 | 0.24566 | 0.6124 m | 16.45° | 4 min |
| E13 | 0.13842 | 0.4056 m | 9.18° | 2 h 58 min |
| E14 | 0.13643 | 0.4017 m | 8.78° | 4 h 29 min |

**Nota:** los errores de posición y orientación corresponden a sus respectivos mínimos durante cada entrenamiento. No deben interpretarse como métricas de un único checkpoint.

## 5.3 Análisis de los experimentos

### E01 y E02: primeras pruebas

Se comprobó el funcionamiento general del pipeline.

E01 permitió verificar la comunicación entre el dataset, el modelo y el ciclo de entrenamiento.

E02 exploró un batch size de 1, pero presentó resultados muy inestables, con errores de posición superiores a 2 metros.

Una posible explicación es el comportamiento de Batch Normalization con lotes extremadamente pequeños, aunque el experimento no permitió aislar esa causa.

### E03: primer resultado competitivo

Se incrementó el entrenamiento a 80 épocas y se utilizó batch size 4.

Se obtuvo:

- Error medio de posición: 38.58 cm.
- Error medio angular: 9.66°.
- Mejor val_loss: 0.13575.

Fue el primer resultado que mostró un comportamiento competitivo para el problema.

### E04–E07: exploración de hiperparámetros

Se estudiaron diferentes tamaños de batch y learning rates.

Los resultados mostraron que:

- Batch size 8 ofrecía un comportamiento prometedor.
- Learning rate 1e-4 presentó mejores resultados que 2e-4.
- Reducir el learning rate a 5e-5 no produjo una mejora consistente.

Esta etapa permitió identificar una configuración base más conveniente.

### E08: mejor configuración de referencia

Se utilizaron:

- Batch size 8.
- Learning rate 1e-4.
- 80 épocas.
- Seed 42.

Resultados:

- Mejor val_loss: 0.118176.
- Error medio de posición: 37.71 cm.
- Error medio angular: 8.81°.

Este entrenamiento se convirtió en el checkpoint de referencia debido a su equilibrio entre las métricas y su menor tiempo de ejecución frente a los entrenamientos prolongados.

### E09: reducción del learning rate

Se mantuvieron las condiciones de E08, pero se redujo el learning rate a 5e-5.

El resultado fue inferior al de E08, con un mejor val_loss de 0.129987.

No se encontró evidencia de que reducir el learning rate mejorara el comportamiento general de esta configuración.

### E10: incremento a 300 épocas

Se mantuvo la configuración de E08 y se incrementó el entrenamiento a 300 épocas.

Se obtuvo:

- Mejor val_loss: 0.118176.
- Mejor error medio de posición: 37.24 cm.
- Mejor error medio angular: 8.81°.

Aunque se consiguió el menor error medio de posición de todos los experimentos, la diferencia respecto a E08 fue de aproximadamente 4.7 mm.

El entrenamiento mostró un deterioro de la validación mientras la pérdida de entrenamiento continuaba disminuyendo, indicando sobreajuste.

Por tanto, el incremento de épocas no representó una mejora general consistente.

### E11: cambio de semilla

Se repitió el entrenamiento de 300 épocas utilizando seed 123.

Resultados:

- Mejor val_loss: 0.120669.
- Error medio de posición: 38.57 cm.
- Mejor error medio angular: 8.22°.

Este experimento permitió observar que la semilla influye en los resultados.

Aunque presentó el menor error angular registrado, también mostró evidencia de sobreajuste.

### E12: smoke test

Se realizó una prueba de dos épocas para comprobar la integración de nuevas funcionalidades:

- Scheduler.
- Early Stopping.
- Weight Decay.

Se denominó smoke test porque su finalidad principal era verificar que los componentes nuevos funcionaran correctamente, no obtener un modelo competitivo.

### E13: Scheduler y Early Stopping

Se incorporó ReduceLROnPlateau para disminuir automáticamente el learning rate cuando la validación dejara de mejorar.

También se implementó Early Stopping para detener el entrenamiento cuando no existiera una mejora suficiente durante un periodo determinado.

El entrenamiento se detuvo en la época 76.

Resultados:

- Mejor val_loss: 0.138419.
- Error medio de posición: 40.56 cm.
- Error medio angular: 9.18°.

Aunque las funcionalidades trabajaron correctamente, el resultado no superó a E08.

### E14: incorporación de Weight Decay

Se mantuvieron los mecanismos de E13 y se agregó:

`weight_decay = 1e-4`

dentro del optimizador Adam.

Resultados:

- Mejor val_loss: 0.136431.
- Error medio de posición: 40.17 cm.
- Mejor error medio angular: 8.78°.

Se obtuvieron mejores medianas angulares y un resultado competitivo, pero no se consiguió eliminar el sobreajuste ni superar el checkpoint de referencia.

# 6. Análisis del sobreajuste

Uno de los principales problemas identificados durante los entrenamientos fue la diferencia entre el comportamiento de entrenamiento y el de validación.

En distintos experimentos se observó que la pérdida de entrenamiento continuaba disminuyendo, mientras que la pérdida de validación dejaba de mejorar o comenzaba a incrementarse.

Esto indica que el modelo conseguía ajustar progresivamente los datos de entrenamiento, pero no mejoraba proporcionalmente su capacidad para estimar poses en imágenes no utilizadas durante el entrenamiento.

Se implementó un sistema de análisis que considera:

- Diferencia entre `train_loss` y `val_loss`.
- Incremento de `val_loss` después de alcanzar su mínimo.
- Evolución de las curvas de aprendizaje.
- Pendiente de las métricas en las últimas épocas.
- Coeficiente de variación para estudiar convergencia.

Se establecieron los siguientes umbrales heurísticos:

| Parámetro | Valor |
|---|---:|
| OVERFIT_GAP_THRESHOLD | 0.05 |
| OVERFIT_RISE_PCT | 5 % |
| UNDERFIT_MIN_LOSS | 0.2 |
| UNDERFIT_DROP_PCT | 10 % |
| CONVERGED_SLOPE_EPS | 0.001 |
| CONVERGED_CV_EPS | 0.02 |

Estos umbrales permiten estandarizar el diagnóstico, pero no constituyen pruebas matemáticas definitivas de sobreajuste o convergencia.

Los experimentos E10, E11 y E14 presentaron evidencia clara de sobreajuste en sus curvas.

E03, E08 y E09 también mostraron señales que justifican revisar sus curvas completas.

# 7. Análisis de los mejores resultados

Hasta el 10 de noviembre de 2026 se identificaron los siguientes resultados destacados:

| Indicador | Resultado | Experimento |
|---|---:|---|
| Menor val_loss | 0.118176 | E08 y E10 |
| Menor error medio de posición | 0.3724 m | E10 |
| Menor error medio angular | 8.2238° | E11 |
| Menor mediana de posición | 35.08 cm | E08 |
| Menor mediana angular | 8.452° | E14 |

Es importante diferenciar entre los mejores resultados individuales y el mejor checkpoint general.

**E08 se mantiene como checkpoint de referencia**, debido a su menor val_loss, buen equilibrio entre posición y orientación y menor coste computacional.

E10 y E11 presentan resultados interesantes en métricas individuales, pero requieren una comparación controlada de sus checkpoints sobre exactamente las mismas imágenes.

Por el momento, ninguno debe declararse ganador absoluto.

# 8. Herramientas de análisis y documentación

Además del entrenamiento, se desarrollaron herramientas para analizar y comparar sistemáticamente los experimentos.

Se utilizó:

`python -m src.analysis_nn.analyze_run`

Para generar reportes individuales de cada entrenamiento.

Los resultados incluyen:

| Archivo | Contenido |
|---|---|
| `01_loss.png` | Curvas de train_loss y val_loss |
| `02_position_error.png` | Error de posición |
| `03_orientation_error.png` | Error angular |
| `04_learning_rate.png` | Evolución del learning rate |
| `05_generalization_gap.png` | Diferencia entre entrenamiento y validación |
| `06_summary.png` | Dashboard general |
| `report.md` | Reporte individual |

También se generaron herramientas de comparación global:

- `analysis/all_reports.md`
- `analysis/summaries/`
- Comparación de val_loss entre entrenamientos.
- Rankings por error de posición y orientación.
- Comparaciones de sobreajuste.
- Análisis de tiempos de ejecución.
- Comparaciones de hiperparámetros.

Esto permitió pasar de evaluar entrenamientos individualmente a contar con un historial experimental reproducible y comparable.

# 9. Limitaciones identificadas

A partir de los experimentos realizados, se identificaron las siguientes limitaciones:

**1. Capacidad de generalización**

Los resultados de validación corresponden a una sola secuencia de video. Todavía no se ha comprobado el comportamiento del modelo sobre el conjunto de prueba independiente.

**2. Sobreajuste**

Incrementar el número de épocas permitió reducir las pérdidas de entrenamiento, pero no produjo mejoras equivalentes en validación.

**3. Limitaciones de la arquitectura**

ResNet-18 fue utilizada como arquitectura inicial para estudiar la viabilidad de la regresión de pose monocular.

Sin embargo, después de múltiples experimentos y ajustes de hiperparámetros, no se consiguió una mejora consistente que justificara continuar optimizando esta misma arquitectura.

Esto no demuestra que ResNet-18 sea incapaz de resolver el problema, pero sí indica que la estrategia actual alcanzó resultados que no mejoraron significativamente con los cambios probados.

**4. Recursos computacionales**

Los entrenamientos se realizaron en CPU, lo que incrementó considerablemente el tiempo necesario para experimentar.

**5. Validación física pendiente**

Aunque las etiquetas se encuentran alineadas al sistema físico, todavía falta comprobar el desempeño del modelo mediante una evaluación independiente y, posteriormente, una integración experimental con el dron.

# 10. Conclusiones de la etapa ResNet-18

Durante esta primera etapa se consiguió:

1. Construir un dataset de imágenes con poses asociadas.
2. Realizar una reconstrucción tridimensional mediante COLMAP.
3. Obtener posiciones y orientaciones de las cámaras.
4. Implementar una alineación métrica entre COLMAP y el sistema físico FLU mediante Umeyama.
5. Generar etiquetas de pose en metros.
6. Implementar un pipeline de entrenamiento y validación en PyTorch.
7. Desarrollar una arquitectura de regresión de pose basada en ResNet-18.
8. Ejecutar y documentar 14 experimentos.
9. Analizar diferentes tamaños de batch, learning rates, semillas y mecanismos de regularización.
10. Desarrollar herramientas para estudiar convergencia, sobreajuste y generalización.

El resultado más competitivo fue E08, con un error medio de posición de aproximadamente 37.71 cm y un error angular medio de 8.81°.

Sin embargo, los experimentos posteriores no consiguieron mejorar consistentemente el comportamiento general del modelo.

Por esta razón, se decidió no continuar realizando ajustes sobre ResNet-18 y comenzar a investigar otras arquitecturas especializadas en estimación de pose.

# 11. Nueva dirección de investigación: PoseNet y sus evoluciones

Durante la reunión del 10 de noviembre de 2026 con los asesores del proyecto se presentaron los resultados obtenidos con ResNet-18 y los diferentes experimentos realizados.

A partir de esta revisión, se recomendó dejar de invertir tiempo en nuevas variaciones de hiperparámetros de ResNet-18 e investigar PoseNet o alguna de sus evoluciones.

La intención es identificar una arquitectura que resulte más apropiada para el problema de estimación de pose de cámara a partir de una imagen monocular.

Las actividades propuestas para esta nueva etapa son:

1. Investigar el artículo original de PoseNet.
2. Revisar los trabajos posteriores que lo citan.
3. Identificar las principales evoluciones y modificaciones de su arquitectura.
4. Comparar sus funciones de pérdida y estrategias de entrenamiento.
5. Identificar implementaciones disponibles en PyTorch.
6. Evaluar cuáles pueden adaptarse a la estimación de pose del Parrot Bebop 2.
7. Implementar la arquitectura seleccionada utilizando el dataset `v2_0`.
8. Comparar sus resultados contra el checkpoint de referencia E08.

Es importante que la nueva arquitectura conserve la compatibilidad con el problema planteado: estimación de posición y orientación 6-DoF de una cámara a partir de una única imagen.

No se busca simplemente sustituir ResNet-18 por otra red neuronal, sino investigar si las modificaciones arquitectónicas y metodológicas desarrolladas para la regresión de pose pueden ofrecer mejores resultados.

---

# 12. Estado general del proyecto al 11 de noviembre de 2026

| Componente | Estado |
|---|---|
| Captura de imágenes | Completado |
| Dataset v2_0 | Completado |
| Reconstrucción COLMAP | Completada |
| Alineación COLMAP → FLU | Implementada |
| Generación de etiquetas 6-DoF | Completada |
| Separación train/validation | Implementada |
| Pipeline PyTorch | Implementado |
| ResNet-18 | Evaluada experimentalmente |
| Análisis de entrenamientos | Implementado |
| Selección definitiva de arquitectura | Pendiente |
| Implementación de PoseNet o evolución | Pendiente |
| Evaluación sobre conjunto de test | Pendiente |
| Comparación física con el dron | Pendiente |
| Integración ROS Noetic | Pendiente |

**Estado actual:** se ha completado una primera etapa de desarrollo y experimentación con ResNet-18. El proyecto cuenta con un dataset reconstruido, etiquetas métricas de pose, un pipeline de aprendizaje profundo y un historial de 14 entrenamientos documentados.

La siguiente etapa se centrará en investigar e implementar una arquitectura derivada de PoseNet que permita continuar con el objetivo de estimación monocular de pose 6-DoF.


**************************************************************************

Perfecto. Con estos informes individuales podemos precisar mucho mejor la evolución de los 14 entrenamientos y, sobre todo, distinguir **qué cambios se introdujeron en el código, qué hipótesis se probaron y qué resultados produjo cada modificación**.

Hay algo importante: los Runs 12, 13 y 14 no son simplemente entrenamientos más largos. Representan una evolución de la estrategia de optimización y regularización.

## 1. Evolución de los 14 entrenamientos

| Run | Modificación o experimento | Resultado principal |
|---|---|---|
| 01 | Primera prueba con ResNet18, batch 2. | Funcionamiento inicial, pero errores elevados. |
| 02 | Batch size 1. | Empeoramiento considerable; entrenamiento inestable. |
| 03 | Batch 4 y 80 épocas. | Primer resultado competitivo: 38.58 cm y 9.66°. |
| 04 | Batch 8 y 10 épocas. | Se comprueba el funcionamiento con lotes mayores. |
| 05 | Batch 4, LR 0.0001. | Comparación con otra configuración de batch. |
| 06 | LR 0.00005. | Se prueba una tasa de aprendizaje menor. |
| 07 | LR 0.0002. | Se prueba una tasa de aprendizaje mayor. |
| 08 | Batch 8, LR 0.0001, 80 épocas. | Primeros resultados destacados: 37.71 cm y 8.81°. |
| 09 | LR 0.00005, 80 épocas. | Resultados inferiores al Run 08. |
| 10 | 300 épocas, batch 8. | Mejor error de posición: 37.24 cm. Se identifica overfitting. |
| 11 | 300 épocas, seed 123. | Mejor error angular: 8.22°. |
| 12 | Smoke test de nuevas funcionalidades. | Se verifica la ejecución del nuevo pipeline. |
| 13 | Scheduler + Early Stopping. | Control automático del LR y detención anticipada. |
| 14 | Scheduler + Early Stopping + Weight Decay. | Se incorpora regularización L2. |

## 2. Qué se fue implementando técnicamente

Esta es la parte más importante para documentar el desarrollo del proyecto.

### Etapa 1. Implementación del modelo de regresión de pose

**Runs 01–03**

Se desarrolló el pipeline inicial basado en ResNet18.

Se implementaron:

- Extracción de características visuales mediante ResNet18 preentrenada.
- Cabeza de regresión para estimar XYZ.
- Cabeza de regresión para estimar orientación mediante cuaterniones.
- Normalización de coordenadas.
- Función de pérdida combinada para posición y orientación.
- Evaluación mediante errores de posición y orientación.

### Etapa 2. Experimentación de hiperparámetros

**Runs 04–09**

Se comenzaron a modificar sistemáticamente los parámetros de entrenamiento:

- Batch size: 4 y 8.
- Learning rate: 0.00005, 0.0001 y 0.0002.
- Número de épocas: 10 y 80.

El resultado más interesante fue identificar que la combinación:

`batch_size=8` y `learning_rate=0.0001`

producía resultados competitivos.

### Etapa 3. Análisis del comportamiento del modelo

**Runs 10–11**

Se incrementó el entrenamiento a 300 épocas.

Esto permitió estudiar:

- Evolución de `train_loss`.
- Evolución de `val_loss`.
- Diferencia entre entrenamiento y validación.
- Errores de posición y orientación por época.
- Influencia de la semilla aleatoria.

Se identificó que el modelo podía continuar aprendiendo sobre el conjunto de entrenamiento mientras empeoraba su rendimiento sobre validación.

Esto permitió detectar el problema de overfitting.

### Etapa 4. Automatización de la optimización

**Runs 12–13**

Se incorporaron mecanismos para mejorar el control del entrenamiento.

**ReduceLROnPlateau**

Permite reducir automáticamente el learning rate cuando la pérdida de validación deja de mejorar.

**Early Stopping**

Permite finalizar el entrenamiento cuando no se consigue una mejora durante un número determinado de épocas.

También se incorporaron parámetros configurables para controlar ambos mecanismos desde la terminal.

### Etapa 5. Regularización

**Run 14**

Se incorporó Weight Decay:

`weight_decay=0.0001`

Esta modificación introduce una penalización L2 sobre los pesos del modelo, buscando reducir su complejidad y mejorar la generalización.

Aunque se observó una mejora en orientación respecto al Run 13, todavía existe sobreajuste.

---

## 3. Comparación de los mejores resultados

Es importante separar los resultados porque cada entrenamiento puede tener su mejor época en momentos diferentes.

| Indicador | Mejor resultado | Entrenamiento |
|---|---:|---|
| Validation Loss | 0.118176 | Run 08 y 10 |
| Error medio de posición | 0.3724 m | Run 10 |
| Error medio angular | 8.2238° | Run 11 |

### Comparación de las últimas tres estrategias

| Métrica | Run 13 | Run 14 |
|---|---:|---:|
| Mejor val_loss | 0.138419 | 0.136431 |
| Mejor posición | 0.4056 m | 0.4017 m |
| Mejor orientación | 9.1771° | 8.7804° |
| Épocas completadas | 76 | 116 |
| Overfitting detectado | No | Sí |
| Weight Decay | No | 0.0001 |

El Run 14 mejoró ligeramente las tres métricas respecto al Run 13, aunque necesitó más tiempo y presentó una brecha mayor entre entrenamiento y validación.

## 4. Conclusión de la evolución

La evolución del proyecto puede resumirse en cuatro grandes objetivos:

1. **Construcción:** desarrollar una red neuronal capaz de estimar la pose 6-DoF a partir de imágenes.
2. **Experimentación:** encontrar hiperparámetros adecuados mediante diferentes configuraciones.
3. **Diagnóstico:** identificar problemas de convergencia y sobreajuste.
4. **Optimización:** incorporar scheduler, early stopping y weight decay para mejorar el entrenamiento.

El avance más importante no es únicamente haber conseguido errores inferiores a 40 cm y 9° de orientación, sino haber desarrollado un pipeline experimental que permite analizar, comparar y controlar sistemáticamente el aprendizaje.

**Mi recomendación:** conservar el Run 10 como referencia de rendimiento, el Run 11 como referencia de orientación y el Run 14 como referencia de la nueva estrategia de regularización. No descartaría ninguno de ellos: cada uno aporta información diferente para decidir la siguiente modificación del modelo.