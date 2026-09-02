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
  remote_setup.sh         # prepara la workstation; no entrena
tests/
REPORT.md                 # decisiones y diferencias frente al paper
```

## Instalación

Se requiere Python 3.12, una versión reciente de `uv` y, para extraer los datos,
7-Zip (`7z`, `7zz` o `7za`).

```bash
uv sync --frozen
uv run animal-audio --help
```

PyTorch está configurado con el índice CUDA 12.6 en `pyproject.toml`, compatible con la workstation usada para los experimentos.

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
```

Genera, entre otros:

- `metrics.json`
- `thresholds.json`
- `thresholds.csv`
- `validation_probabilities.csv`
- `precision_recall.png`
- `prevalence_vs_ap.png`

Se reportan resultados con umbral fijo 0.5, como en el paper, y con umbrales
optimizados exclusivamente sobre validación.

## Predicciones de test

Con umbral fijo 0.5:

```bash
uv run animal-audio predict \
  --config configs/nddr_pcen.yaml \
  --checkpoint artifacts/experiments/nddr_pcen/best.pt \
  --device cuda
```

Con umbrales de validación:

```bash
uv run animal-audio predict \
  --config configs/nddr_pcen.yaml \
  --checkpoint artifacts/experiments/nddr_pcen/best.pt \
  --thresholds artifacts/experiments/nddr_pcen/thresholds.json \
  --device cuda
```

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
