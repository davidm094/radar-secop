"""Deja lista la app sin entrenar nada: copia los datos congelados y descarga el modelo y el índice ya construidos.

Lo usa app/app.py al arrancar. Sirve para desplegar la demo directo desde el repo (Streamlit Community Cloud) y
para montarla en un computador en un par de minutos. Los archivos se publican en un Release de GitHub porque
pesan demasiado para el repo; los genera scripts/17_empaquetar_demo.py a partir de los pasos 15 y 07.
Si los archivos ya existen (porque se corrieron los scripts), no descarga nada.
"""
import os
import shutil
import threading

from .config import DATA_PROC, DATA_RAW, MODELS, MODELO_APP, ROOT

RELEASE = os.environ.get("RADAR_RELEASE_URL", "https://github.com/davidm094/radar-secop/releases/download/demo-v2/")
INDICE = ["procesos_idx.parquet", "procesos_tfidf.joblib", "procesos_emb.npy"]
TAMANOS = {"procesos_idx.parquet": 49794787, "procesos_tfidf.joblib": 43540925, "procesos_emb.npy": 152036480}   # bytes, Release demo-v2


def _descargar(nombre: str, destino, aviso):
    import requests
    aviso(f"Descargando {nombre}…")
    tmp = destino.with_suffix(destino.suffix + ".part")
    with requests.get(RELEASE + nombre, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for bloque in r.iter_content(1 << 20):
                f.write(bloque)
    tmp.rename(destino)


SPACY_WHL = ("https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/"
             "es_core_news_sm-3.7.0-py3-none-any.whl")


def _asegurar_spacy(aviso):
    """El modelo de spaCy 3.7.0 es con el que se prepararon los textos (el 3.8.0 lematiza distinto), y su paquete
    no se puede poner en un requirements.txt junto a spaCy 3.8. Si no está instalado, se baja y se descomprime."""
    import importlib.util
    import zipfile
    destino = MODELS / "es_core_news_sm"
    if importlib.util.find_spec("es_core_news_sm") or destino.exists():
        return
    import requests
    aviso("Descargando el modelo de spaCy…")
    whl = MODELS / "es_core_news_sm.whl"
    whl.write_bytes(requests.get(SPACY_WHL, timeout=120).content)
    with zipfile.ZipFile(whl) as z:
        z.extractall(MODELS, [n for n in z.namelist() if n.startswith("es_core_news_sm/")])
    whl.unlink()


_CANDADO = threading.Lock()   # dos visitantes a la vez no deben descargar lo mismo


def asegurar_artefactos(aviso=print):
    with _CANDADO:
        _asegurar(aviso)


def _asegurar(aviso):
    _asegurar_spacy(aviso)
    cong = ROOT / "data" / "congelado"
    for nombre, carpeta in (("contratos_prep.parquet", DATA_PROC), ("unspsc_v14.parquet", DATA_RAW)):
        if not (carpeta / nombre).exists():
            shutil.copy(cong / nombre, carpeta / nombre)
    modelo = MODELS / f"{MODELO_APP}.joblib"
    if not modelo.exists():
        _descargar(modelo.name, modelo, aviso)
    # El índice son tres archivos que deben venir del mismo origen. Si falta alguno y los que hay son los del
    # Release (mismo tamaño), es una descarga interrumpida y se completa; si son otros, los generó el paso 07.
    hay = [n for n in INDICE if (DATA_PROC / n).exists()]
    del_release = all((DATA_PROC / n).stat().st_size == TAMANOS[n] for n in hay)
    if len(hay) < len(INDICE) and not del_release:
        raise FileNotFoundError("El índice de procesos está incompleto en data/processed/. Corra scripts/07_indexar_procesos.py "
                                "o borre procesos_idx.parquet, procesos_tfidf.joblib y procesos_emb.npy para descargarlo.")
    for n in INDICE:
        if n not in hay:
            _descargar(n, DATA_PROC / n, aviso)
