# Reconocimiento multietiqueta de animales con NDDR-MTL

Implementación desde cero en PyTorch de **NDDR-MTL** basada en el paper
*How to analyse overlapping sounds in the marine environment using supervised
multi-label classification*.

El proyecto implementa solamente la propuesta NDDR-MTL solicitada. No incluye
CMLC, Binary Relevance ni HPS-MTL.

## Componentes

- Lectura y validación de `train.csv` y WAV PCM.
- Split 80/20 reproducible con semilla 42 y aislamiento por grabación.
- Frontend MS-PCEN adaptado a audio de 22.05 kHz.
- NDDR-MTL con 42 ramas específicas, tres fusiones NDDR y conexiones skip.
- Tres BiGRU por especie, temporal max pooling y 42 logits.
- Entrenamiento con BCE multietiqueta, `pos_weight`, AMP y checkpoints.
- Métricas del paper, métricas F1/mAP, concurrencia y baseline nulo.
- Optimización opcional de umbrales usando únicamente validación.
- Ablación MS-PCEN frente a log-Mel.
- Predicciones de test en CSV.

## Estructura

```text
configs/
  nddr_pcen.yaml          # experimento principal
  nddr_logmel.yaml        # ablación de la representación
src/animal_audio/
  config.py               # configuración estricta
  data.py                 # WAV, metadata y split por grupos
  features.py             # Mel, PCEN y log-Mel
  model.py                # implementación NDDR-MTL
  metrics.py              # métricas y gráficos
  engine.py               # entrenamiento, evaluación e inferencia
  cli.py                  # comandos
scripts/
  generate_analysis.py        # comparación, gráficos y análisis de errores
  generate_audio_examples.py  # visualiza forma de onda, log-Mel y PCEN
  remote_setup.sh             # prepara la workstation; no entrena
results/                      # métricas y figuras finales versionadas
presentation/
  audio/                      # clip WAV usado en la apertura de Reveal.js
  figures/audio_examples/     # ejemplos reales para reporte y diapositivas
  styles.scss                 # estilos de la presentación Reveal.js
tests/
REPORT.qmd                # reporte Quarto reproducible de la implementación
REPORT.html               # versión HTML renderizada y autocontenida
PRESENTATION.qmd          # diapositivas Reveal.js reproducibles
PRESENTATION.html         # presentación renderizada y autocontenida
```

## Instalación

Se requiere Python 3.12, una versión reciente de `uv` y, para extraer los datos,
7-Zip (`7z`, `7zz` o `7za`).

```bash
uv sync --frozen
uv run animal-audio --help
```

PyTorch está configurado con el índice CUDA 12.6 en `pyproject.toml`,
compatible con la workstation usada para los experimentos.

## Renderizar el reporte

Con Quarto instalado, el reporte reproducible se genera mediante:

```bash
quarto render REPORT.qmd --to html
```

El comando produce `REPORT.html` como un documento autocontenido, incluyendo
las fórmulas MathJax y el diagrama Mermaid. La presentación se genera con:

```bash
quarto render PRESENTATION.qmd --to revealjs
```

Esto produce `PRESENTATION.html`, también autocontenido.

## Preparar los datos

Los archivos suministrados deben estar en la raíz:

```text
train.7z
test.7z
train.csv
```

Extraer y crear el split reproducible:

```bash
uv run animal-audio prepare --config configs/nddr_pcen.yaml --extract
```

Si los archivos ya están extraídos como `train/*.wav` y `test/*.wav`:

```bash
uv run animal-audio prepare --config configs/nddr_pcen.yaml
```

Para recrear el split:

```bash
uv run animal-audio prepare --config configs/nddr_pcen.yaml --force
```

El split se guarda en `artifacts/split_seed42.csv`. Las ventanas solapadas de
una grabación permanecen juntas para evitar fuga de información.

## Generar ejemplos de las representaciones

El siguiente comando selecciona de forma determinística un clip sin especies,
uno con una especie y uno con múltiples especies:

```bash
uv run python scripts/generate_audio_examples.py \
  --config configs/nddr_pcen.yaml \
  --output-dir artifacts/figures/audio_examples \
  --examples-per-category 1 \
  --seed 42
```

Cada PNG compara la forma de onda, log-Mel y PCEN del mismo clip. También se
genera `manifest.csv` con el archivo, categoría y etiquetas. Estas figuras son
solo para visualización; el entrenamiento sigue calculando PCEN directamente en
memoria.

## Inspeccionar el modelo sin entrenar

```bash
uv run animal-audio inspect-model \
  --config configs/nddr_pcen.yaml \
  --device cuda \
  --dry-forward
```

El preset compacto de 42 tareas contiene aproximadamente 500 mil parámetros.

## Entrenamiento

> Este es el único comando que modifica los pesos del modelo.

Experimento principal:

```bash
uv run animal-audio train \
  --config configs/nddr_pcen.yaml \
  --device cuda
```

Ablación log-Mel, usando el mismo split y presupuesto:

```bash
uv run animal-audio train \
  --config configs/nddr_logmel.yaml \
  --device cuda
```

Antes de ejecutar estos comandos se debe revisar en la workstation:

- GPU y memoria disponibles.
- `batch_size` y `num_workers`.
- Espacio para datos y checkpoints.
- Que `artifacts/split_seed42.csv` sea idéntico para ambos experimentos.

Para reanudar, establecer `training.resume_from` en el YAML con la ruta a
`last.pt` y volver a ejecutar `train`.

## Evaluación

```bash
uv run animal-audio evaluate \
  --config configs/nddr_pcen.yaml \
  --checkpoint artifacts/experiments/nddr_pcen/best.pt \
  --device cuda

uv run animal-audio evaluate \
  --config configs/nddr_logmel.yaml \
  --checkpoint artifacts/experiments/nddr_logmel/best.pt \
  --device cuda
```

Genera, entre otros:

- `metrics.json`
- `thresholds.json`
- `thresholds.csv`
- `validation_probabilities.csv`
- `precision_recall.png`
- `prevalence_vs_ap.png`

Se reportan resultados con umbral fijo 0.5, como en el paper, y con umbrales
optimizados exclusivamente sobre validación. Para generar la comparación,
curvas, análisis por especie, concurrencia y errores:

```bash
uv run python scripts/generate_analysis.py \
  --pcen-dir artifacts/experiments/nddr_pcen \
  --logmel-dir artifacts/experiments/nddr_logmel \
  --split-csv artifacts/split_seed42.csv \
  --config configs/nddr_logmel.yaml \
  --output-dir results/analysis \
  --threshold 0.5
```

Los resultados compactos seleccionados para la entrega están en `results/`.

## Predicciones de test

Con umbral fijo 0.5:

```bash
uv run animal-audio predict \
  --config configs/nddr_logmel.yaml \
  --checkpoint artifacts/experiments/nddr_logmel/best.pt \
  --device cuda
```

Los umbrales optimizados por clase se conservan como artefacto de análisis, pero
no se utilizaron para el test porque degradaron micro F1 y produjeron exact
match cero en la misma validación.

Salidas:

- `test_probabilities.csv`
- `test_predictions.csv`

Ambas conservan `filename` seguido de las 42 especies en el orden de
`train.csv`.

## Workstation remota

Después de autorizar una llave SSH, copiar el código:

```bash
ssh <usuario>@<host> "mkdir -p ~/animal-audio-nddr"
rsync -av --partial --progress \
  --exclude .venv --exclude artifacts --exclude train --exclude test \
  ./ <usuario>@<host>:~/animal-audio-nddr/
```

Copiar los datos por separado para poder reanudar la transferencia:

```bash
rsync -av --partial --progress train.7z test.7z train.csv \
  <usuario>@<host>:~/animal-audio-nddr/
```

En la workstation:

```bash
cd ~/animal-audio-nddr
bash scripts/remote_setup.sh
uv run animal-audio prepare --config configs/nddr_pcen.yaml --extract
uv run animal-audio inspect-model --config configs/nddr_pcen.yaml --device cuda
```

`scripts/remote_setup.sh` no inicia entrenamiento.

## Pruebas

```bash
uv run python -m unittest discover -s tests -v
```

Las pruebas usan datos sintéticos y no entrenan el dataset real.
