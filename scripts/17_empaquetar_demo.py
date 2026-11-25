"""Empaqueta el modelo y el índice de la app para publicarlos en un Release de GitHub (ver src/radar/demo.py).

Correr después de los pasos 15 y 07. Deja en dist/ cuatro archivos:
  familia_v2.joblib       el modelo de la app con los pesos en 32 bits y comprimido (60 MB en vez de 134)
  procesos_idx.parquet, procesos_tfidf.joblib, procesos_emb.npy   el índice de procesos, tal como lo deja el paso 07
Antes de escribir comprueba que el modelo en 32 bits predice lo mismo que el original en todo el test:
misma familia en cada contrato y las mismas alertas.
"""
import copy
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
import pandas as pd

from radar.config import DATA_PROC, MODELO_APP, ROOT
from radar.demo import INDICE
from radar.model import cargar

DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)
m = cargar(MODELO_APP)
m32 = copy.deepcopy(m)
clf = m32.named_steps["clf"]
clf.coef_, clf.intercept_ = clf.coef_.astype("float32"), clf.intercept_.astype("float32")

df = pd.read_parquet(DATA_PROC / "contratos_prep.parquet")
ids = set(pd.read_csv(DATA_PROC / "test_ids.csv").iloc[:, 0])
te = df[df.id_contrato.isin(ids) & df.fam.isin(m.classes_)]
P, P32 = m.predict_proba(te.txt_lem), m32.predict_proba(te.txt_lem)
alerta = lambda Q: (Q.max(1) >= 0.70) & (m.classes_[Q.argmax(1)] != te.fam.values)
assert (P.argmax(1) == P32.argmax(1)).all() and (alerta(P) == alerta(P32)).all(), "el modelo en 32 bits no predice igual"
print(f"Modelo en 32 bits: misma familia en {len(te):,} contratos de test, mismas {int(alerta(P).sum()):,} alertas, "
      f"diferencia máxima de probabilidad {np.abs(P - P32).max():.1e}")
joblib.dump(m32, DIST / f"{MODELO_APP}.joblib", compress=3)
for n in INDICE:
    shutil.copy(DATA_PROC / n, DIST / n)
for f in sorted(DIST.iterdir()):
    print(f"{f.stat().st_size / 1e6:7.1f} MB  {f.name}")
