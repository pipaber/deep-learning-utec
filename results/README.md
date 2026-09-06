# Resultados versionados

Estos artefactos se generaron con los mejores checkpoints de cada experimento
sobre el mismo split de validación 80/20 y con umbral de decisión 0.5.

## Resumen

<!-- markdownlint-disable MD013 MD060 -->

| Sistema | Mejor epoch | mAP | Micro F1 | Macro F1 | Exact match |
|---------|-------------:|----:|---------:|---------:|------------:|
| Baseline de ceros | — | 0.04143 | 0.00000 | 0.00000 | 0.34667 |
| NDDR-MTL + PCEN | 85 | 0.50641 | 0.66684 | 0.44231 | 0.38412 |
| NDDR-MTL + log-Mel | 23 | **0.55069** | **0.69797** | **0.46905** | **0.43561** |

<!-- markdownlint-enable MD013 MD060 -->

El mAP y las métricas macro excluyen clases sin positivos en validación. Los
resultados con umbrales optimizados se conservan en cada `metrics.json`. En
log-Mel mejoraron macro F1 de 0.46905 a 0.58483, pero redujeron micro F1 de
0.69797 a 0.41339 y produjeron exact match cero; por eso no son el resultado
principal.

## Contenido

- `analysis/summary.json`: resumen y diferencias log-Mel menos PCEN.
- `analysis/per_class_metrics.csv`: AP, F1 y prevalencia por clase.
- `analysis/error_rankings.csv`: tasas de falsos positivos y negativos.
- `analysis/label_combination_errors.csv`: errores por combinación verdadera.
- `analysis/*.png`: curvas y visualizaciones incluidas en reporte/presentación.
- `pcen/` y `logmel/`: métricas, historias y figuras originales de evaluación.
- `test_prediction_summary.json`: modelo y umbral usados para test.

Las tablas completas `test_probabilities.csv` y `test_predictions.csv`, junto
con el checkpoint final, permanecen en `artifacts/final/` porque son artefactos
grandes ignorados por Git.

## Reproducción

```bash
uv run python scripts/generate_analysis.py \
  --pcen-dir artifacts/experiments/nddr_pcen \
  --logmel-dir artifacts/experiments/nddr_logmel \
  --split-csv artifacts/split_seed42.csv \
  --config configs/nddr_logmel.yaml \
  --output-dir results/analysis \
  --threshold 0.5
```
