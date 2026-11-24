"""Control del umbral de alertas en un conjunto de validación.

El umbral de 0,70 se eligió mirando el test (scripts/04), y sobre ese mismo test se cuentan las alertas. Para
comprobar que esa elección no es optimista, aquí se aparta un 10 % del entrenamiento como validación (agrupado
por texto, con la distribución real), se entrena con el resto y se repite la tabla del umbral en la validación.
Si la validación y el test dan lo mismo, el umbral no está sobreajustado al test.
Salida: reports/validacion_umbral.json. No cambia el modelo de la app.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from radar.config import DATA_PROC, REPORTS, ROOT, SEED
from radar.labels import particion_por_texto, clases_validas, balancear, excluir_gold
from radar.model import construir, cargar

df = pd.read_parquet(DATA_PROC / "contratos_prep.parquet")
clases = clases_validas(df, "fam")
df = df[df.fam.isin(clases)]
tr, te = particion_por_texto(df, "txt_norm")
tr = excluir_gold(tr, df, ROOT / "data" / "gold" / "gold_ids.csv")
i_tr, i_val = next(GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED).split(tr, groups=tr.txt_norm))
tr2, val = tr.iloc[i_tr], tr.iloc[i_val]
assert not set(val.txt_norm) & set(tr2.txt_norm) and not set(val.txt_norm) & set(te.txt_norm)


def tabla(m, d):
    P = m.predict_proba(d.txt_lem.tolist())
    p, c, y = m.classes_[P.argmax(1)], P.max(1), d.fam.values
    r = {"n": int(len(d)), "accuracy": round(float((p == y).mean()), 3),
         "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 3),
         "top3": round(float(np.mean([a in b for a, b in zip(y, m.classes_[np.argsort(-P, 1)[:, :3]])])), 3), "umbral": {}}
    for u in (0.5, 0.6, 0.7, 0.8, 0.9):
        k = c >= u
        r["umbral"][str(u)] = {"cobertura": round(float(k.mean()), 3), "acuerdo_con_SECOP": round(float((p[k] == y[k]).mean()), 3),
                               "tasa_de_alertas": round(float((k & (p != y)).mean()), 4),
                               "alertas_desde_8011": round(float((y[k & (p != y)] == "8011").mean()), 3)}
    return r


b2 = balancear(tr2, "fam")
m2 = construir().fit(b2.txt_lem, b2.fam)
out = {"entrenamiento_sin_validacion": int(len(b2)), "validacion": tabla(m2, val), "test_con_ese_mismo_modelo": tabla(m2, te),
       "test_con_el_modelo_de_la_app": tabla(cargar("familia_v1"), te)}
(REPORTS / "validacion_umbral.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
for k in ("validacion", "test_con_ese_mismo_modelo", "test_con_el_modelo_de_la_app"):
    print(k, {x: out[k][x] for x in ("n", "accuracy", "macro_f1", "top3")}, out[k]["umbral"]["0.7"], out[k]["umbral"]["0.9"])
