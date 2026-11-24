"""Control del umbral de alertas en un conjunto de validación.

El umbral de 0,70 se eligió mirando el test (scripts/04), y sobre ese mismo test se cuentan las alertas. Para
comprobar que esa elección no es optimista, aquí se aparta un 10 % del entrenamiento como validación (agrupado
por texto, con la distribución real), se entrena con el resto y se repite la tabla del umbral en la validación.
Si la validación y el test dan lo mismo, el umbral no está sobreajustado al test.
El control se hace dos veces: con las etiquetas de SECOP (como se eligió el umbral) y con las etiquetas de 8011
corregidas por Claude (como el modelo de la app). En la segunda, del 8011 de entrenamiento solo se usan los
contratos que Claude re-etiquetó y que no cayeron en la validación.
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

from radar.config import DATA_PROC, REPORTS, ROOT, SEED, MODELO_APP, MODELO_SECOP
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
       "test_con_el_modelo_de_etiquetas_secop": tabla(cargar(MODELO_SECOP), te)}

# Lo mismo con las etiquetas de 8011 corregidas por Claude
c = pd.DataFrame([json.loads(l) for l in (ROOT / "data" / "llm" / "train_claude_8011.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]).drop_duplicates("id")
c = c[~c.no_clasificable & c.familia.isin(clases)]
mapa = dict(zip(c.id, c.familia))
d2 = tr2[(tr2.fam != "8011") | tr2.id_contrato.isin(mapa)].copy()
d2["fam"] = d2.id_contrato.map(mapa).fillna(d2.fam)
b3 = balancear(d2, "fam")
m3 = construir().fit(b3.txt_lem, b3.fam)
out["etiquetas_corregidas"] = {"entrenamiento_sin_validacion": int(len(b3)),
                               "reetiquetados_usados": int(d2.id_contrato.isin(mapa).sum()),
                               "validacion": tabla(m3, val), "test_con_ese_mismo_modelo": tabla(m3, te),
                               "test_con_el_modelo_de_la_app": tabla(cargar(MODELO_APP), te)}
(REPORTS / "validacion_umbral.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
for g, o in (("secop", out), ("corregidas", out["etiquetas_corregidas"])):
    for k in [x for x in o if x.startswith(("validacion", "test_"))]:
        print(g, k, {x: o[k][x] for x in ("n", "accuracy", "macro_f1", "top3")}, o[k]["umbral"]["0.7"], o[k]["umbral"]["0.9"])
