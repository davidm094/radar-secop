# Radar SECOP

**Clasificación explicable de contratos públicos colombianos en familias UNSPSC, para que las pymes encuentren procesos de SECOP II y para revisar la calidad de los códigos declarados.**

Proyecto final de Procesamiento de Lenguaje Natural, Pontificia Universidad Javeriana.
Autores: David Moreno y Dierick Brochero.

| Entregable | Dónde está |
|---|---|
| Documento final (13 páginas, marco de Ashmore et al., 2021) | [`documento_final.pdf`](documento_final.pdf) |
| Pipeline y código | `scripts/` (un paso por script) y `src/radar/` |
| Demo | `streamlit run app/app.py` (ver "Cómo correrlo") |
| Resultados | `reports/` |

## Problema y usuario

Toda compra pública en Colombia se publica en SECOP II con un código del clasificador UNSPSC (estándar de la ONU adoptado por Colombia Compra Eficiente). Para una pyme, ese código es la puerta de entrada: si no sabe en qué familia cae lo que vende, no encuentra los procesos que le sirven. El catálogo tiene 420 familias y está escrito en lenguaje de catálogo, no en el de un empresario.

Radar SECOP recibe la descripción del negocio **en palabras de la pyme** y devuelve:

1. las 3 familias UNSPSC más probables, con la explicación de por qué (contribución exacta de cada palabra y contratos reales parecidos);
2. los procesos publicados en SECOP II que mejor le encajan (búsqueda híbrida semántica + léxica + familia).

Un segundo usuario, veedurías y equipos de control interno, usa el mismo modelo para marcar contratos cuyo código declarado no coincide con lo que dice su objeto. Esos contratos quedan **para revisar**: una discrepancia no es un error ni una irregularidad.

**Tarea de PLN:** clasificación de texto multiclase (175 familias con datos suficientes), más recuperación de información y explicabilidad.

## Hallazgo central: las etiquetas de SECOP están sesgadas

La familia 8011 (Servicios de recursos humanos) se usa como cajón de sastre para casi todo contrato de prestación de servicios. La guía oficial de codificación (G-CBS-02, 2026, Caso 10) dice que un servicio profesional debe codificarse según el servicio que recibe la entidad: un abogado va en 8012, Servicios legales. En los datos, en cambio:

- en el conjunto de test, el 83 % de los contratos con "abogado" están en 8011 y solo el 9 % en 8012;
- en el análisis exploratorio, frases de trámite como "por sus propios medios" predicen 8011 con P = 0,97 (la tasa base es 0,68), y 188 de 1.139 entidades ponen el 95 % o más de sus contratos en el segmento 80.

Esto cambia el diseño del proyecto:

- Las métricas contra SECOP miden **acuerdo con SECOP, no exactitud**.
- Para la evaluación se arma un **gold set** de 300 contratos con una guía basada en el Caso 10: 215 con etiqueta humana y los 300 con dos anotaciones independientes de Claude. La confiabilidad de cada fuente se mide y se reporta (ver Resultados).
- Las fórmulas de contratación se quitan del texto para que el modelo no aprenda el atajo hacia 8011.

## Pipeline

```
 SECOP II (datos.gov.co, API SODA)        Catálogo UNSPSC v14 en español (Colombia Compra Eficiente)
          │                                              │
 01 descarga ─────────────────────────────────────────────┘
          │  contratos 2025–2026 (objeto + código declarado) · procesos publicados/abiertos
 02 preparación de texto
          │  normalizar (minúsculas, sin tildes ni códigos) → quitar fórmulas de contratación
          │  → lematizar con spaCy es_core_news_sm, sin stopwords
          │  variante A = solo normalizado · variante B = sin fórmulas + lematizado (la de la app)
 03 etiquetas + modelo
          │  familia = 4 primeros dígitos del código · familias con ≥ 30 ejemplos (175)
          │  split 80/20 agrupado por texto · tope de 3.000 por familia
          │  el gold set se excluye del entrenamiento (por id y por texto idéntico)
          │  TF-IDF (1–2 gramas, 200k términos) + regresión logística uno-contra-resto (SGD, log-loss)
 04 explicabilidad: global por familia · local exacta (tf-idf × coeficiente) · contrafactuales
          │  · vecinos más cercanos · fidelidad ERASER contra referencia al azar · calibración del umbral
 05 sesgo por entidad: split por entidad (entidades nunca vistas)
 06 discrepancias: código declarado ≠ predicción con confianza ≥ 0,70 → "para revisar"
 07 índice de procesos: texto + familia (inferida si viene sin código) + embeddings model2vec
 08 gold set (planillas de anotación): acuerdo entre anotadores, desacuerdos, plantilla de resolución
 09 variante con datos corregidos por Claude (solo si kappa Claude–gold ≥ 0,61; no se cumplió)
 10 evaluación de la búsqueda: consultas escritas por personas, juicios a ciegas, P@5 y nDCG@10
 11 revisión humana de 100 alertas de discrepancia
 12 clave del gold set (código declarado de cada contrato, que los anotadores no ven)
 13 evaluación final contra el gold set: confiabilidad de cada fuente · error de SECOP por estrato
          │  · modelo vs. referencia humana y vs. consenso de Claude · plausibilidad de la explicación
 14 prueba con textos estilo pyme generados por Claude (resultado negativo; no cambia el modelo)
 15 destilación: entrenar con las etiquetas de 8011 corregidas por Claude (mejora contra el gold; la app no cambia)
 16 control del umbral de alertas en un conjunto de validación aparte
          │
 app/ Streamlit: "Para pymes" · "Revisar un contrato" · "Alertas para veedurías" · "Cómo funciona"
```

### Ejemplo real de punta a punta

| Paso | Valor |
|---|---|
| Objeto original | "Prestación de servicios profesionales como abogado para el acompañamiento jurídico de la personería municipal" |
| Código declarado | 80111600 → familia **8011** Servicios de recursos humanos |
| Sin fórmulas + lematizado | `abogado acompanamiento juridico personeria municipal` |
| Predicción (variante B) | **8012** Servicios legales, P = 0,73 |
| Explicación | "jurídico" (+2,77) y "abogado" (+1,78) son los términos que más empujan hacia 8012 |
| Contrafactual | sin "abogado jurídico", el modelo cae a 8011 con P = 0,16: sin el oficio, solo queda el cajón de sastre |
| App | se marca **para revisar** (8011 declarado ≠ 8012 sugerido con confianza ≥ 0,70) |

## Resultados

Contra las etiquetas de SECOP (acuerdo, no exactitud). Familia (175 clases). El test natural tiene la distribución real (63 % de 8011); el balanceado tiene como máximo 300 por familia.

| Variante | Test | Accuracy | Macro-F1 | Top-3 | ECE |
|---|---|---|---|---|---|
| Siempre la clase mayoritaria | natural | 0,629 | 0,004 | – | – |
| A: normalizado (con fórmulas) | natural | 0,413 | 0,307 | 0,712 | 0,085 |
| B: sin fórmulas + lematizado | natural | 0,397 | 0,302 | 0,704 | 0,096 |
| A: normalizado (con fórmulas) | balanceado | 0,559 | 0,408 | 0,751 | 0,091 |
| B: sin fórmulas + lematizado | balanceado | 0,559 | 0,407 | 0,759 | 0,074 |
| B, entidades nunca vistas | balanceado | 0,428 | 0,293 | 0,649 | 0,040 |

Cómo leer la tabla:

- La accuracy de "siempre 8011" es alta por el desbalance, pero su macro-F1 es casi cero. Por eso se reporta macro-F1 y top-3.
- Quitar las fórmulas (B) casi no cambia el acuerdo con SECOP. Lo que cambia es **qué aprende el modelo**: en A, entre los términos más fuertes de 8011 están "servicios profesionales" (peso 3,07) y "prestar servicios" (2,96). En B esas frases desaparecen, aunque aparecen atajos nuevos de oficina y lugar ("secretaría", "gobernación"); se documenta como limitación.
- Con entidades nunca vistas, el macro-F1 balanceado baja de 0,41 a 0,29: parte de lo que el modelo aprende son costumbres de cada entidad.
- **Oficios:** en test, SECOP pone al 83 % de los abogados en 8011; el modelo B predice 8012 para el 59 %. En enfermería, SECOP pone el 51 % en 8510 y el modelo B el 78 %.
- **Explicabilidad (ERASER, 500 contratos de test):** quitar las 3 palabras que el modelo señala baja la probabilidad de la clase 0,29, frente a 0,09 si se quitan 3 al azar (comprehensiveness). Las palabras que el modelo señala sí son las que usa.
- **Umbral de discrepancia:** con confianza ≥ 0,9 el modelo coincide con SECOP en el 95 % de los casos, pero cubre solo el 1,2 % de los contratos. Con ≥ 0,7, cubre el 14 % y coincide en el 68 %. Con el umbral de 0,70 se marcan 1.640 de 36.844 contratos de test (4,5 %), y el 80 % de ellos están declarados en 8011.

### Evaluación contra el gold set (paso 13)

El gold set tiene 300 contratos fuera del entrenamiento. 215 tienen etiqueta humana (un anotador, Dierick Brochero), 53 de ellos tienen además la etiqueta de un segundo anotador (David Moreno, columna `humano2`; anotó consultando a Claude como buscador del catálogo, y el acuerdo entre los dos anotadores humanos en 49 contratos comparables es 22,4 %, kappa 0,20) y los 300 tienen dos anotaciones independientes de Claude (Sonnet 5.5 y Fable 5.2), hechas sin ver el código de SECOP. Todas las etiquetas están en `data/gold/gold_anotaciones.csv`.

**Confiabilidad.** Las dos anotaciones de Claude coinciden entre sí en el 72,6 % de los contratos (kappa 0,72). Las etiquetas humanas coinciden con cada una de ellas en el 27 % (kappa 0,25–0,26) y con SECOP en el 14 %. En un control ciego de 60 contratos, 24 de los 47 desacuerdos son etiquetas humanas que no encajan con la guía (`reports/gold_control_ciego_revision.csv`; esa revisión la hizo Claude, que es parte interesada; el segundo anotador humano revisó 17 de esos 24 contratos y no coincidió con la primera etiqueta en ninguno). Por eso se reporta contra dos referencias: **H** (humana, 215) y **C** (consenso de las dos anotaciones de Claude, 208). Ninguna es una verdad de referencia limpia: H es ruidosa y C no es humana y excluye los casos ambiguos.

| Medida | H: humana (215) | C: consenso de Claude (208) | H y C coinciden (48) |
|---|---|---|---|
| SECOP distinto de la referencia (familia) | 85,6 % | 73,1 % | 62,5 % |
| … en contratos declarados en 8011 | 98,7 % | 97,6 % | 94,7 % |
| Modelo B: accuracy | 0,228 | 0,481 | 0,625 |
| Modelo B: top-3 | 0,405 | 0,755 | 0,854 |
| Modelo B: macro-F1 | 0,110 | 0,359 | 0,449 |

- Lo firme: casi todo lo declarado en 8011 en la muestra corresponde a otra familia, con cualquier referencia.
- Lo abierto: la exactitud del modelo a nivel de familia está entre 0,23 y 0,48 según la referencia.
- Plausibilidad de la explicación (F1 de palabras frente a las palabras clave humanas): 0,287 vs. 0,173 al azar.
- Kappa Claude–humano = 0,25, por debajo de la compuerta de 0,61 que se había fijado para el paso 09. Como ese gold humano resultó poco confiable, la corrección de etiquetas se probó con otra regla, fijada antes de correr (ver "Entrenar con las etiquetas de 8011 corregidas").
- Los textos estilo pyme de `data/llm/aumentado.jsonl` no dependen de esa compuerta y se probaron aparte (`scripts/14_prueba_textos_pyme.py` → `reports/prueba_textos_pyme.json`): el macro-F1 balanceado sube de 0,407 a 0,424, el gold queda igual (0,481 → 0,476 contra el consenso de Claude) y con las consultas de pyme la familia del proceso relevante queda en el top-3 en 56,8 % frente a 60,4 % sin ellos. Es un resultado negativo: el modelo de la app no cambió.

### Entrenar con las etiquetas de 8011 corregidas (paso 15)

Claude Sonnet 5.5 re-etiquetó, con la guía del gold set y sin ver el código declarado, los 3.000 contratos de entrenamiento declarados en 8011 (`data/llm/train_claude_8011.jsonl`): dejó 36 en 8011, marcó 148 como no clasificables y mandó el resto sobre todo a 8016, 8012 y 9314. Regla fijada antes de correr: el modelo nuevo (D) debe superar a B frente a la anotación ciega de Claude Fable sobre el gold (otro modelo y otra fecha que el maestro), sin perder más de 2 puntos de macro-F1 contra SECOP fuera de 8011.

| Referencia | B (app) | D (8011 corregido) | Solo acierta D / solo B |
|---|---|---|---|
| Gold, Claude Fable (283): accuracy | 0,431 | 0,488 | 20 / 4 |
| Gold, consenso de Claude (208): accuracy | 0,481 | 0,548 | 18 / 4 |
| Gold, humana (215): accuracy | 0,228 | 0,270 | 11 / 2 |
| SECOP sin 8011, test balanceado: macro-F1 | 0,411 | 0,417 | – |

D cumple la regla y mejora contra las tres referencias, incluida la humana (prueba de signos: p = 0,002, 0,004 y 0,02). Salvedades: el maestro y dos de los árbitros son modelos de Claude, y D casi nunca predice 8011, así que su acuerdo con SECOP en el test natural baja de 0,397 a 0,216 y sus alertas suben de 1.640 a 3.004. **La app sigue usando B**, porque la revisión de alertas y la evaluación de la búsqueda se hicieron con B. Detalle en `reports/destilacion_8011.json`.

### Umbral de alertas en validación (paso 16)

El umbral de 0,70 se eligió mirando el test. Al repetir la cuenta apartando el 10 % del entrenamiento como validación (14.737 contratos), la tasa de alertas es 4,7 % y el acuerdo con SECOP de las predicciones confiadas 67,3 %; en el test son 4,5 % y 68,3 %. El umbral no está sobreajustado al test (`reports/validacion_umbral.json`).

### BETO frente al modelo lineal (175 clases, mismo split)

| Modelo | Test | Accuracy | Macro-F1 | Top-3 |
|---|---|---|---|---|
| TF-IDF + regresión logística (B) | balanceado | 0,559 | 0,407 | 0,759 |
| BETO afinado (3 épocas, T4) | balanceado | 0,524 | 0,262 | 0,730 |
| TF-IDF + regresión logística (B) | natural | 0,397 | 0,302 | 0,704 |
| BETO afinado (3 épocas, T4) | natural | 0,459 | 0,221 | 0,794 |

BETO no supera al modelo lineal en macro-F1 y mejora la accuracy y el top-3 solo en el test natural; con pocas épocas aprende sobre todo las clases grandes.

### Búsqueda (30 consultas escritas por una persona, 1.002 juicios a ciegas)

| Sistema | P@5 | P@10 | nDCG@10 |
|---|---|---|---|
| Familia | 0,293 | 0,303 | 0,307 |
| Léxico (TF-IDF) | 0,520 | 0,557 | 0,610 |
| Semántico (model2vec) | 0,560 | 0,553 | 0,588 |
| Híbrido (el de la app) | 0,587 | 0,553 | 0,597 |

El híbrido gana en P@5; en nDCG@10 el léxico queda ligeramente por encima, así que la diferencia entre los tres no es concluyente con 30 consultas.

### Alertas de discrepancia (100 casos con confianza ≥ 0,70, un revisor)

La familia sugerida por el modelo era la correcta en el 89 % de los casos (SECOP 5 %, ninguna 3 %, no clasificable 3 %). Es la precisión de la alerta en la zona de alta confianza, no en todos los contratos.

## Instalación

Probado con Python 3.13, scikit-learn 1.9 y spaCy 3.8; se recomienda Python 3.13 para obtener las mismas cifras.

```bash
git clone https://github.com/davidm094/radar-secop && cd radar-secop
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt   # versiones exactas usadas (o requirements.txt para las últimas)
pip install --no-deps https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0-py3-none-any.whl
```

Se fija el modelo de spaCy 3.7.0 porque es con el que se calcularon las cifras (`--no-deps` evita que pip baje spaCy a 3.7; con spaCy 3.8 sale un aviso de versión, pero funciona).

## Cómo correrlo

**Para reproducir exactamente las cifras del documento** (recomendado), use la foto de datos del 14-oct-2026
que trae el repo y sáltese los pasos 01 y 02 (SECOP cambia a diario, así que descargar de nuevo da otras cifras):

```bash
python scripts/preparar_demo.py             # hace los tres pasos de abajo solo si faltan (unos 25 min la primera vez)
streamlit run app/app.py
```

o paso a paso:

```bash
python scripts/00_usar_datos_congelados.py  # copia data/congelado/ a data/processed/ y data/raw/
python scripts/03_entrenar_evaluar.py       # ~5 min; test balanceado B: macro-F1 0,407, top-3 0,759
python scripts/07_indexar_procesos.py       # ~20 min
streamlit run app/app.py
```

Para descargar datos nuevos desde cero:

```bash
python scripts/01_descargar.py              # ~300k contratos 2025+, procesos abiertos y catálogo (20–40 min)
python scripts/02_preparar.py               # limpieza y lematización
python scripts/03_entrenar_evaluar.py       # modelos A y B + métricas + auditoría del atajo
python scripts/04_explicabilidad.py
python scripts/05_particion_entidad.py
python scripts/06_discrepancias.py
python scripts/07_indexar_procesos.py       # índice para la búsqueda
streamlit run app/app.py                    # demo en http://localhost:8501
```

Con el gold set anotado y, opcionalmente, con Claude:

```bash
python scripts/08_acuerdo_gold.py --desacuerdos                 # inconsistencias de anotación, sin abrir la clave
python scripts/13_referencias_gold.py                           # evaluación final contra el gold (dos referencias)
python scripts/14_prueba_textos_pyme.py                         # prueba: entrenar con los textos estilo pyme (no cambia el modelo de la app)
python scripts/15_destilacion_8011.py                           # entrenar con las etiquetas de 8011 corregidas por Claude
python scripts/16_validacion_umbral.py                          # control del umbral de alertas en validación
python scripts/11_revisar_discrepancias.py evaluar              # precisión de la alerta (revisión humana)
python llm/reetiquetar.py --entrada data/gold/gold_set_plantilla.xlsx --salida data/llm/gold_claude.jsonl
python scripts/09_entrenar_con_llm.py                           # solo corre si kappa Claude–gold ≥ 0,61
python scripts/10_evaluar_busqueda.py pool                      # luego marcar relevancia a mano
python scripts/10_evaluar_busqueda.py evaluar
```

Pruebas básicas (limpieza de texto, gold fuera del entrenamiento, coherencia de los reportes): `python tests/test_basico.py`.

Los modelos y el índice no se suben al repositorio porque pesan más de 100 MB; se regeneran con los pasos 03 y 07. Los resultados (métricas, explicaciones, discrepancias) quedan en `reports/`.

## Estructura

```
src/radar/      config · data (descarga SODA y catálogo) · text (limpieza y lematización) · labels (niveles,
                split, balanceo, exclusión del gold) · model · evaluate (métricas y ECE) · explain · search
scripts/        00–16, un paso del pipeline cada uno (00 = datos congelados, 11 = revisión de discrepancias,
                12 = clave del gold set, 13 = evaluación final contra el gold con dos referencias,
                14 = prueba con textos estilo pyme, 15 = destilación de 8011, 16 = validación del umbral)
app/app.py      demo en Streamlit
llm/            re-etiquetado y aumento con Claude Code (ver llm/README.md)
notebooks/      beto_colab.ipynb: BETO afinado en Colab (GPU) con el mismo split que el modelo lineal
data/beto/      split congelado (train balanceado + test) para que la comparación con BETO sea justa
data/gold/      gold set: ids, plantilla de anotación, clave y todas las etiquetas con su procedencia (gold_anotaciones.csv)
data/consultas/ consultas de pyme, juicios de relevancia y rankings de la evaluación de búsqueda
data/congelado/ foto de los datos del 14-oct-2026 para reproducir las cifras
reports/        métricas y salidas de cada paso
tests/          pruebas básicas (test_basico.py) y simulador de Claude para probar llm/ sin gastar cuota
```

## Uso de LLM

Claude **nunca** predice en la app. Se usó para tres cosas, todas declaradas:

1. anotar dos veces los 300 contratos del gold set con la misma guía que los anotadores humanos y sin ver el código declarado (Sonnet 5.5 y Fable 5.2). Esas anotaciones son la segunda referencia de la evaluación, reportada siempre aparte de la humana;
2. re-etiquetar los 3.000 contratos de entrenamiento declarados en 8011 (paso 15). Mejora el modelo contra el gold set, pero la app sigue con el modelo original;
3. generar 7.001 descripciones de negocio estilo pyme: se probaron (paso 14) y no mejoraron la búsqueda, así que el modelo de la app no las usa.

Detalle, comandos y controles en [`llm/README.md`](llm/README.md).

## Limitaciones conocidas

- Las etiquetas de entrenamiento del modelo de la app son las de SECOP, con el sesgo descrito.
- Se usan 300.000 contratos de 1,9 millones, los primeros que entrega la API y no una muestra aleatoria; 315 de las 358 familias tienen menos de 300 contratos.
- El gold set es pequeño (300) y su parte humana resultó poco confiable (kappa 0,20 entre los dos anotadores humanos); la referencia alternativa no es humana. Por eso la exactitud del modelo se reporta como un rango.
- Las 30 consultas de búsqueda y la revisión de 100 alertas las hizo una sola persona.
- El lematizador de spaCy comete errores en algunas palabras ("área" → "areo"); se usa el modelo `es_core_news_sm` 3.7 con spaCy 3.8.
- Al quitar las fórmulas aparecen atajos de oficina y municipio; con entidades nuevas el desempeño cae.
- El catálogo en español disponible es v14; la guía 2026 cita UNSPSC v26, publicado solo en inglés.
- Los procesos "abiertos" son una foto del día de descarga.

## Datos y licencias

- SECOP II – Contratos Electrónicos (`jbjy-vk9h`) y Procesos de Contratación (`p6dx-8zbt`), datos.gov.co, licencia CC BY-SA 4.0.
- Clasificador de Bienes y Servicios v14 en español, Colombia Compra Eficiente.
- Guía de Codificación de Bienes y Servicios G-CBS-02 (2026), Colombia Compra Eficiente.
- Embeddings: `minishlab/potion-multilingual-128M` (model2vec).
