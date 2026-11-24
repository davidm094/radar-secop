"""Rutas y constantes del proyecto Radar SECOP."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# Conjuntos de datos abiertos (datos.gov.co, API SODA, licencia CC BY-SA 4.0)
SODA_BASE = "https://www.datos.gov.co/resource"
DS_CONTRATOS = "jbjy-vk9h"   # SECOP II - Contratos Electrónicos
DS_PROCESOS = "p6dx-8zbt"    # SECOP II - Procesos de Contratación

# Catálogo UNSPSC v14 en español, publicado por Colombia Compra Eficiente
UNSPSC_XLS_URL = (
    "https://operaciones.colombiacompra.gov.co/sites/cce_public/files/"
    "cce_clasificador/clasificador_de_bienes_y_servicios_v14_1.xls"
)

SEED = 42
MIN_EJEMPLOS_CLASE = 30      # familias con menos ejemplos se descartan
MAX_EJEMPLOS_CLASE = 3000    # tope por familia en entrenamiento (balanceo)
# Modelos guardados en models/
MODELO_SECOP = "familia_v1"  # entrenado con las etiquetas de SECOP tal como vienen (scripts/03)
MODELO_APP = "familia_v2"    # el de la app: igual, pero las etiquetas de 8011 del entrenamiento las corrigió Claude (scripts/15)
TOP_K = 3                    # sugerencias que ve la pyme
UMBRAL_DISCREPANCIA = 0.70   # confianza mínima para marcar una discrepancia (scripts/06 y la app)

for p in (DATA_RAW, DATA_PROC, MODELS, REPORTS / "figs"):
    p.mkdir(parents=True, exist_ok=True)
