# Guion del video explicativo

**Proyecto:** Reconocimiento multietiqueta de animales con NDDR-MTL  
**Duración objetivo:** 17 minutos y 30 segundos  
**Límite:** 18 minutos

> Las indicaciones entre corchetes no se narran. Los tiempos son acumulados y
> sirven como referencia. Las diapositivas del apéndice se reservan para
> preguntas.

---

# 1. Metodología — 0:00 a 6:00

## Slide: Primero, escuchemos el problema

**Tiempo: 0:00–0:35**

[Reproducir el audio antes de continuar.]

“Comencemos escuchando un clip real de tres segundos grabado en la Amazonía.

En este audio aparecen dos especies, identificadas en el dataset como `DENMIN`
y `LEPPOD`. El desafío no consiste en escoger una única respuesta, sino en
reconocer todas las especies que pueden aparecer simultáneamente. Por eso,
nuestro problema es de clasificación multietiqueta.”

## Slide: Del sonido a 42 decisiones

**Tiempo: 0:35–1:00**

“Para cada clip, el modelo produce un vector de 42 posiciones, una por especie.
Cada posición representa una decisión binaria de presencia o ausencia.

Un audio puede contener cero, una o varias especies; por tanto, no es una
clasificación multiclase, porque las categorías no son mutuamente excluyentes.”

## Slide: La escala del reto

**Tiempo: 1:00–1:25**

“Trabajamos con audio mono de tres segundos, muestreado a 22.05 kilohertz. Cada
archivo contiene 66,150 muestras.

El conjunto etiquetado tiene 62,191 clips: 49,721 para entrenamiento y 12,470
para validación. Además, existen 31,187 clips de test sin etiquetas, para los
que debemos producir 42 probabilidades y 42 decisiones.”

## Slide: ¿Cómo es realmente el dataset?

**Tiempo: 1:25–1:55**

“El análisis exploratorio muestra un fuerte desbalance entre especies. Aunque
la mediana es una especie por clip, existen clips con hasta ocho especies y
16,210 ejemplos contienen tres o más etiquetas.

También encontramos coocurrencias frecuentes. Por ejemplo, `SPHSUR` y `BOABIS`
aparecen juntas 8,582 veces. Esto indica que las tareas no son independientes y
motiva una arquitectura capaz de compartir información entre especies.”

## Slide: El dilema: ¿compartir o separar?

**Tiempo: 1:55–2:25**

“Existen dos estrategias extremas. En Binary Relevance, o BR, cada clase tiene
una red totalmente independiente. Esto permite especialización, pero no
aprovecha relaciones entre clases.

En el extremo opuesto, CMLC utiliza una única representación para todas las
clases. Es más eficiente, pero puede obligar a especies diferentes a utilizar
las mismas características.

NDDR-MTL busca un punto intermedio: conserva rutas específicas, pero permite un
intercambio aprendido de información.”

## Slide: Arquitecturas evaluadas en el paper

**Tiempo: 2:25–3:05**

“Nuestra metodología se basa en el paper de Olcay y colaboradores, publicado en
2026, sobre clasificación multietiqueta de sonidos superpuestos en ambientes
marinos.

El paper compara una CRNN convencional, Binary Relevance, Hard Parameter
Sharing y NDDR-MTL. En HPS-MTL se comparte el extractor convolucional y luego se
separan las rutas recurrentes. En NDDR-MTL se mantienen rutas específicas y se
introducen capas NDDR y conexiones shortcut.

El paper trabaja con cinco tareas bioacústicas. Nosotros adaptamos la
arquitectura NDDR-MTL a 42 especies amazónicas.”

## Slide: Primero hacemos visible el sonido

**Tiempo: 3:05–3:35**

“El primer componente convierte la onda en una representación tiempo-frecuencia.
Aplicamos una STFT con FFT de 512 y salto de 128, seguida de un banco de 64
bandas Mel. Esto produce 513 frames temporales.

En el modelo principal aplicamos PCEN, que atenúa variaciones lentas de energía
y busca resaltar eventos acústicos sobre el ruido de fondo. En la ablación
reemplazamos PCEN por log-Mel.”

## Slide: El mismo clip, ahora visible

**Tiempo: 3:35–3:55**

“Aquí observamos el mismo clip como forma de onda, log-Mel y PCEN. PCEN hace más
visibles algunos eventos locales y reduce cambios lentos de energía.

Estas imágenes solo sirven para visualizar la transformación. El modelo recibe
el tensor numérico completo, no el archivo PNG.”

## Slide: Luego modelamos 42 tareas que colaboran

**Tiempo: 3:55–4:30**

“La red contiene 42 rutas, una por especie. Cada ruta tiene tres bloques CNN,
tres BiGRU y un clasificador propio.

Después de cada bloque convolucional colocamos una capa NDDR. Así tenemos tres
puntos de intercambio de información. También utilizamos shortcuts desde las
tres profundidades para conservar tanto características tempranas como
profundas.”

## Slide: Cada especie aprende sus propios filtros

**Tiempo: 4:30–4:55**

“Cada bloque convolucional utiliza una convolución de cinco por cinco, Batch
Normalization, ReLU, Max Pooling y Dropout.

Los pesos de las 42 ramas son independientes, de modo que cada especie aprende
sus propios filtros. Las convoluciones agrupadas únicamente permiten ejecutar
las ramas juntas de forma más eficiente en la GPU; no implican compartir
pesos.”

## Slide: La solución: intercambio NDDR — Paso 1

**Tiempo: 4:55–5:15**

“En el primer paso de NDDR reunimos las representaciones de todas las tareas.
Cada especie produce cuatro canales. Al concatenar las 42 ramas obtenemos 168
canales, sin mezclarlos todavía.”

## Slide: La solución: intercambio NDDR — Paso 2

**Tiempo: 5:15–5:30**

“En el segundo paso aplicamos Batch Normalization sobre los 168 canales. Esto
estabiliza la representación conjunta antes del intercambio de información.”

## Slide: La solución: intercambio NDDR — Paso 3

**Tiempo: 5:30–6:05**

“Finalmente, cada especie tiene su propia convolución de uno por uno. Esta
proyección aprende cómo combinar los 168 canales y vuelve a producir cuatro
canales específicos para la tarea.

La convolución uno por uno no altera las dimensiones de frecuencia y tiempo:
solo aprende qué información tomar de cada especie. Esta es la contribución
central de NDDR-MTL: compartir conocimiento sin eliminar la especialización.”

---

# 2. Implementación — 6:05 a 11:55

## Slide: La pista inicial: confiar en la propia tarea

**Tiempo: 6:05–6:35**

“Las proyecciones NDDR no comienzan de forma aleatoria. Inicialmente asignamos
un peso de 0.6 a las características de la propia especie y distribuimos el 0.4
restante entre las otras 41.

Cada conexión externa comienza aproximadamente en 0.0098. Es solo una
inicialización: durante el entrenamiento, Adam puede fortalecer, reducir o
invertir cualquiera de estas relaciones.”

## Slide: No olvidar las primeras pistas

**Tiempo: 6:35–7:05**

“Las salidas de los tres niveles NDDR tienen distintas resoluciones de
frecuencia. Redimensionamos las dos primeras, concatenamos las tres y aplicamos
una nueva fusión de uno por uno.

Los pesos iniciales son 0.2, 0.2 y 0.6. Así enfatizamos la representación más
profunda sin descartar patrones detectados en etapas anteriores.”

## Slide: De patrones temporales a 42 decisiones

**Tiempo: 7:05–7:40**

“Después de la CNN, cada especie procesa sus 513 frames con tres BiGRU. Cada
dirección utiliza 16 unidades y las dos direcciones se promedian.

Luego aplicamos max pooling temporal y una capa lineal para producir un logit
por especie. Los 42 logits se apilan en un tensor de tamaño batch por 42.

Entrenamos con `BCEWithLogitsLoss`, que incorpora sigmoid de forma numéricamente
estable. Durante inferencia aplicamos sigmoid para obtener probabilidades.”

## Slide: Hacer viable el salto de 5 a 42 especies

**Tiempo: 7:40–8:35**

“La principal dificultad fue escalar una arquitectura de cinco tareas a 42.
Las capas NDDR crecen rápidamente porque cada tarea recibe características de
todas las demás.

Por eso conservamos la topología del paper, pero reducimos su ancho. Pasamos de
64 a cuatro canales CNN y de 128 a 16 unidades GRU por dirección. En total,
tenemos 42 ramas convolucionales y 126 BiGRU específicas. El modelo resultante
tiene 500,682 parámetros.

También adaptamos el procesamiento de señal. El paper trabaja a 96 kilohertz
con FFT de 2048 y salto de 512. Nuestro dataset está a 22.05 kilohertz, por lo
que usamos FFT de 512 y salto de 128 para mantener una escala temporal
aproximadamente equivalente.

El modelo recibe tres canales, pero en nuestro caso son tres copias del mismo
espectrograma. Esto mantiene la interfaz arquitectónica, aunque no aporta
información adicional.”

## Slide: Validar sin escuchar dos veces lo mismo

**Tiempo: 8:35–9:25**

“Los clips son ventanas extraídas de grabaciones más largas. Algunas ventanas
consecutivas comparten dos de sus tres segundos.

Una división aleatoria por clip podría colocar audio casi idéntico en
entrenamiento y validación, causando fuga de información y métricas demasiado
optimistas.

Para evitarlo, identificamos la grabación original y realizamos una división
por grupos. Evaluamos 4,096 candidatos con `GroupShuffleSplit` y elegimos un
split 80-20 que mantuviera soporte por especie. Utilizamos la semilla 42 y
verificamos que ninguna grabación original apareciera en ambos conjuntos.”

## Slide: Cómo enseñamos a la red

**Tiempo: 9:25–10:55**

“Entrenamos en una NVIDIA RTX A2000 de 12 gigabytes, usando PyTorch 2.11 y CUDA
12.6. Utilizamos Adam, batch size 64 y 100 épocas sin early stopping. Activamos
precisión mixta en CUDA para reducir memoria y acelerar el entrenamiento.

La pérdida binaria está ponderada para compensar parcialmente el desbalance.
También usamos gradient clipping con valor 5. El weight decay general fue de
10 a la menos 4 y las proyecciones NDDR utilizaron 10 a la menos 2 para
regularizar el intercambio entre tareas. Guardamos el último checkpoint y el
de mejor mAP.

El learning rate comenzó en 0.001 y se multiplicó por 0.75 cada 388 pasos. Como
una época tiene aproximadamente 777 pasos, el decay se aplicó dos veces por
época.

Esta configuración fue demasiado agresiva. Cerca de la época 23, el learning
rate ya era aproximadamente 1.8 por 10 a la menos 9, por lo que el modelo casi
había dejado de aprender. Una mejora futura sería aplicar el mismo decay cada
diez épocas, aproximadamente cada 7,770 pasos.”

## Slide: ¿Qué estamos comparando?

**Tiempo: 10:55–11:55**

“Comparamos tres sistemas.

El primero es un baseline que siempre predice ausencia para las 42 especies. No
es una red neuronal; es un control para medir qué ocurre si explotamos la gran
cantidad de etiquetas negativas.

El segundo es NDDR-MTL con PCEN, nuestra implementación principal del método del
paper.

El tercero es NDDR-MTL con log-Mel, utilizado como ablación de la representación
de entrada.

Los dos modelos NDDR comparten arquitectura, split, semilla, optimizador, batch
y 100 épocas. La única diferencia experimental es PCEN frente a log-Mel. Por
eso, la comparación busca aislar el efecto de la representación.”

---

# 3. Resultados — 11:55 a 17:30

## Slide: ¿Qué aprendió el modelo?

**Tiempo: 11:55–12:45**

“El mejor resultado corresponde a NDDR-MTL con log-Mel, usando el checkpoint de
la época 23.

Sobre los 12,470 clips de validación obtiene un mAP de 0.55069, micro F1 de
0.69797, macro F1 de 0.46905 y exact match de 0.43561. El Hamming loss es
0.02912 y utilizamos un umbral global de 0.5.

Micro F1 reúne todas las decisiones y está más influenciado por especies
frecuentes. Macro F1 calcula el resultado por especie y les da el mismo peso.
Exact match exige acertar las 42 decisiones del clip simultáneamente.”

## Slide: Comparación final

**Tiempo: 12:45–13:15**

“El baseline de ceros obtiene un mAP de 0.04143 y un micro F1 de cero. Sin
embargo, alcanza un exact match cercano a 0.347 porque acierta los clips sin
especies etiquetadas. Esto muestra por qué exact match no debe interpretarse de
forma aislada.

Los dos modelos NDDR superan claramente este control, confirmando que aprenden
patrones de presencia y no solo la ausencia dominante.”

## Slide: ¿Cuándo dejó de aprender?

**Tiempo: 13:15–13:45**

“Las curvas muestran que ambos modelos convergen temprano. El learning rate cae
casi a cero mucho antes de la época 100, por lo que continuar el entrenamiento
aporta muy pocos cambios.

Esto limita la comparación: los resultados son válidos para la configuración
utilizada, pero un scheduler más lento podría mejorar ambos modelos.”

## Slide: La pregunta experimental: ¿PCEN ayuda?

**Tiempo: 13:45–14:35**

“PCEN obtiene 0.50641 de mAP y 0.66684 de micro F1. Log-Mel alcanza 0.55069 y
0.69797, respectivamente.

Log-Mel mejora el mAP en 0.04428, aproximadamente 8.7 por ciento de forma
relativa. También mejora el exact match de 0.38412 a 0.43561.

Por tanto, PCEN no mejora el rendimiento agregado de este dataset bajo nuestra
configuración. Sin embargo, el efecto depende de la especie. Log-Mel favorece
especialmente a `SCIALT`, `BOAPRA` y `SCIRIZ`, mientras que PCEN funciona mejor
para `LEPNOT` y `RHIICT`. La conclusión no es que PCEN siempre sea peor, sino
que su beneficio no fue uniforme ni positivo en promedio.”

## Slide: El reto aumenta con la concurrencia

**Tiempo: 14:35–15:00**

“A medida que aumenta el número de especies presentes, el problema se vuelve
más difícil, especialmente para exact match.

Con más especies existen más oportunidades de producir una omisión o una
inserción. Aunque el modelo acierte la mayoría de las etiquetas, un único error
invalida el clip completo para esta métrica.”

## Slide: ¿Dónde se equivoca?

**Tiempo: 15:00–15:30**

“Las mayores tasas de falsos negativos aparecen principalmente en clases muy
raras, por lo que los valores de 100 por ciento deben interpretarse según su
soporte.

Un caso más preocupante es `PHYNAT`: el modelo omite 203 positivos, equivalentes
al 95.31 por ciento de sus apariciones en validación. Esto puede reflejar falta
de ejemplos, confusión acústica o una limitación de la arquitectura compacta.”

## Slide: Errores de mayor confianza

**Tiempo: 15:30–16:10**

“También inspeccionamos los errores con mayor confianza. Dos falsos positivos
pertenecen a ventanas consecutivas de la misma grabación. Estas ventanas
comparten dos segundos y, en ambas, el modelo predice `SCIFUV` con probabilidad
cercana a 0.994, aunque no está etiquetada.

La persistencia del error sugiere una confusión sistemática con otro sonido.
También podría existir una omisión de anotación. Sin embargo, el espectrograma
y la predicción no bastan para decidir la causa; sería necesario revisar el
audio con un especialista.”

## Slide: Limitaciones

**Tiempo: 16:10–16:50**

“Las principales limitaciones son las siguientes: utilizamos un solo split y
una sola semilla, por lo que no tenemos intervalos de confianza. Tres clases no
tienen positivos en validación y el dataset está fuertemente desbalanceado.

El scheduler fue demasiado agresivo y la arquitectura tuvo que reducirse a
cuatro canales CNN y 16 unidades GRU por dirección. Además, los tres canales de
entrada contienen el mismo espectrograma.

Finalmente, test no tiene etiquetas, por lo que podemos generar predicciones,
pero no calcular métricas finales sobre ellas.”

## Slide: Del modelo a 31,187 predicciones

**Tiempo: 16:50–17:10**

“Utilizamos el mejor checkpoint log-Mel, correspondiente a la época 23, para
procesar los 31,187 audios de test.

Guardamos las 42 probabilidades por clip y luego aplicamos el umbral 0.5 para
obtener decisiones binarias. Conservamos las probabilidades para poder cambiar
el umbral sin repetir la inferencia.”

## Slide: Lo que construimos

**Tiempo: 17:10–17:30**

“En conclusión, implementamos NDDR-MTL con 42 rutas específicas, intercambio
aprendido mediante convoluciones uno por uno, shortcuts entre profundidades y
modelado temporal con BiGRU.

La ablación mostró que log-Mel superó a PCEN, con 0.55069 frente a 0.50641 de
mAP. El trabajo demuestra que es posible adaptar NDDR-MTL de cinco a 42 tareas,
pero también que la representación, el desbalance y la configuración del
entrenamiento son tan importantes como la propia arquitectura.”

---

# Apéndices — solo para preguntas

No narrar durante el video principal:

- `Apéndice: AP por especie`
- `Apéndice: combinaciones frecuentes`
- `Apéndice: distribución de clases`
- `Apéndice: dimensiones completas`
- `Apéndice: pérdida multietiqueta`
- `Apéndice: código de la implementación`

# Referencia principal

Olcay et al. (2026). *How to analyse overlapping sounds in the marine
environment using supervised multi-label classification*.
<https://doi.org/10.1038/s44384-026-00060-x>
