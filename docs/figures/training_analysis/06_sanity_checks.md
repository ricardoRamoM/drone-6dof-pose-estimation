## Sanity check E08 vs E10

- Tramo comparado: épocas 1–80.
- Máxima diferencia absoluta en val_loss: **0.000e+00**.
- Veredicto: **OK — idénticos (misma seed, misma config)**.
- A partir de la época 81, E10 sigue: `val_loss` final = 0.1865 (debería ser > mínimo).

## Sanity check E13 / E14 (scheduler + early stop)

### E13

- Épocas reales: **76**
- No se pudo leer el número de épocas configuradas del history; verificar manualmente.
- Curva de LR encontrada en key `lr`.
- Caídas de LR detectadas en épocas: **[35, 46, 57, 68]**.
- LR inicial: 1.00e-04 → final: 6.25e-06.

### E14

- Épocas reales: **116**
- No se pudo leer el número de épocas configuradas del history; verificar manualmente.
- Curva de LR encontrada en key `lr`.
- Caídas de LR detectadas en épocas: **[28, 54, 78, 89, 100, 111]**.
- LR inicial: 1.00e-04 → final: 1.56e-06.
