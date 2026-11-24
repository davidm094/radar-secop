"""Destilación: ¿mejora el modelo si las etiquetas de 8011 del entrenamiento las corrige Claude?

El sesgo de SECOP está en la familia 8011. Aquí Claude Sonnet 5.5 re-etiqueta, con la misma guía y el mismo
prompt que usó en el gold set (y sin ver el código declarado), los 3.000 contratos de entrenamiento declarados
en 8011 (data/llm/train_claude_8011.jsonl). El resto del entrenamiento conserva la etiqueta de SECOP. El test y
el gold set nunca llevan etiquetas de Claude como entrada.

Variantes (mismo modelo y misma partición que scripts/03):
  B   etiquetas de SECOP tal como vienen (models/familia_v1, scripts/03).
  D   las etiquetas de 8011 se reemplazan por las de Claude; se descartan los "no clasificable" y las
      familias fuera de las 175 del modelo. Es el modelo de la app: se guarda como models/familia_v2.
  D2  como D, pero solo con las etiquetas de confianza alta o media (análisis secundario).

Regla fijada antes de ver el resultado (24-nov-2026): D se considera mejora si
  1. supera a B en acuerdo con la anotación ciega de Claude Fable 5.2 sobre el gold set (otro modelo, otra fecha), y
  2. no pierde más de 2 puntos de macro-F1 contra SECOP en las familias distintas de 8011 (test balanceado).
Salvedad: el árbitro del punto 1 es otro modelo de Claude, así que la evaluación no es independiente del maestro.
Como la regla se cumplió, la app pasó a usar D (24-nov-2026). Claude no predice en la app: solo corrigió
etiquetas de entrenamiento. Salida: reports/destilacion_8011.json y models/familia_v2.joblib.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from radar.config import DATA_PROC, REPORTS, ROOT, MODELO_APP
from radar.labels import particion_por_texto, clases_validas, balancear, excluir_gold
from radar.model import construir, guardar
from radar.text import preparar

UMBRAL = 0.70
df = pd.read_parquet(DATA_PROC / "contratos_prep.parquet")
clases = clases_validas(df, "fam")
df = df[df.fam.isin(clases)]
tr, te = particion_por_texto(df, "txt_norm")
tr = excluir_gold(tr, df, ROOT / "data" / "gold" / "gold_ids.csv")
te_bal = balancear(te, "fam", tope=300)
base = balancear(tr, "fam")[["id_contrato", "txt_lem", "fam"]]

ruta = ROOT / "data" / "llm" / (sys.argv[1] if len(sys.argv) > 1 else "train_claude_8011.jsonl")
c = pd.DataFrame([json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]).drop_duplicates("id")
assert not set(c.id) & set(te.id_contrato), "hay contratos de test entre los re-etiquetados"
assert set(c.id) <= set(base[base.fam == "8011"].id_contrato), "hay contratos que no son de 8011 de entrenamiento"
util = c[~c.no_clasificable & c.familia.isin(clases)]
mapa = dict(zip(util.id, util.familia))
mapa2 = dict(zip(util[util.confianza.isin(["alta", "media"])].id, util[util.confianza.isin(["alta", "media"])].familia))


def corregir(m: dict) -> pd.DataFrame:
    d = base.copy()
    es = d.id_contrato.isin(set(c.id))                 # contratos de 8011 que Claude revisó
    d.loc[es, "fam"] = d.loc[es, "id_contrato"].map(m)  # sin etiqueta útil -> NaN -> se descarta
    return d.dropna(subset=["fam"])


variantes = {"B": base, "D": corregir(mapa), "D2": corregir(mapa2)}

gold = pd.read_csv(ROOT / "data" / "gold" / "gold_anotaciones.csv", dtype=str).fillna("")
gtxt = [preparar(t) for t in gold.objeto_del_contrato]
cons = (gold.claude_sonnet == gold.claude_fable) & ~gold.claude_sonnet.isin(["NC", ""])
refs = {"claude_fable": (~gold.claude_fable.isin(["NC", ""]), gold.claude_fable),
        "claude_sonnet": (~gold.claude_sonnet.isin(["NC", ""]), gold.claude_sonnet),
        "C_consenso_claude": (cons, gold.claude_sonnet),
        "H_humano": (~gold.humano.isin(["NC", ""]), gold.humano),
        "HC_humano_y_consenso": (cons & (gold.humano == gold.claude_sonnet), gold.humano),
        "humano2": (gold.humano2 != "", gold.humano2),
        "secop": (gold.secop != "", gold.secop)}

out = {"reetiquetados": int(len(c)), "no_clasificable": int(c.no_clasificable.sum()),
       "fuera_de_las_175_familias": int((~c.no_clasificable & ~c.familia.isin(clases)).sum()),
       "se_quedan_en_8011": int((util.familia == "8011").sum()),
       "confianza": c.confianza.value_counts().to_dict(),
       "destino_top10": util.familia.value_counts().head(10).to_dict(), "variantes": {}}
pred_gold, modelos = {}, {}
for nombre, d in variantes.items():
    m = construir().fit(d.txt_lem, d.fam)
    modelos[nombre] = m
    r = {"n_entrenamiento": int(len(d)), "n_8011_en_entrenamiento": int((d.fam == "8011").sum())}
    for conj, t in [("test_natural", te), ("test_balanceado", te_bal)]:
        P = m.predict_proba(t.txt_lem.tolist())
        p = m.classes_[P.argmax(1)]
        y = t.fam.values
        no = y != "8011"
        r[conj] = {"accuracy": round(float((p == y).mean()), 3),
                   "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 3),
                   "top3": round(float(np.mean([a in b for a, b in zip(y, m.classes_[np.argsort(-P, 1)[:, :3]])])), 3),
                   "macro_f1_sin_8011": round(float(f1_score(y[no], p[no], average="macro", zero_division=0,
                                                              labels=sorted(set(y[no])))), 3),
                   "accuracy_sin_8011": round(float((p[no] == y[no]).mean()), 3),
                   "predice_8011": round(float((p == "8011").mean()), 3)}
        if conj == "test_natural":
            al = (P.max(1) >= UMBRAL) & (p != y)
            r["alertas_test"] = {"n": int(al.sum()), "tasa": round(float(al.mean()), 3),
                                 "desde_8011": round(float((y[al] == "8011").mean()), 3)}
    Pg = m.predict_proba(gtxt)
    top = m.classes_[np.argsort(-Pg, 1)[:, :3]]
    pred_gold[nombre] = top[:, 0]
    r["gold"] = {}
    for k, (msk, ref) in refs.items():
        y = ref[msk].values
        t3 = top[msk.values]
        r["gold"][k] = {"n": int(msk.sum()), "accuracy": round(float((t3[:, 0] == y).mean()), 3),
                        "top3": round(float(np.mean([a in b for a, b in zip(y, t3)])), 3),
                        "segmento": round(float(np.mean([a[:2] == b[0][:2] for a, b in zip(y, t3)])), 3)}
    out["variantes"][nombre] = r
    print(nombre, json.dumps(r, ensure_ascii=False))

# Comparación pareada B contra D en el gold: ¿cuántos contratos acierta uno y falla el otro?
out["pareado_B_vs_D"] = {}
for k, (msk, ref) in refs.items():
    y, pb, pd_ = ref[msk].values, pred_gold["B"][msk.values], pred_gold["D"][msk.values]
    solo_d, solo_b = int(((pd_ == y) & (pb != y)).sum()), int(((pb == y) & (pd_ != y)).sum())
    n = solo_d + solo_b
    from math import comb
    p_val = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(solo_d, solo_b) + 1)) / 2 ** n) if n else 1.0
    out["pareado_B_vs_D"][k] = {"solo_acierta_D": solo_d, "solo_acierta_B": solo_b, "p_signos": round(p_val, 4)}
print(json.dumps(out["pareado_B_vs_D"], ensure_ascii=False))
b, d = out["variantes"]["B"], out["variantes"]["D"]
out["regla"] = {"1_supera_a_B_contra_claude_fable": d["gold"]["claude_fable"]["accuracy"] > b["gold"]["claude_fable"]["accuracy"],
                "2_no_pierde_mas_de_2_puntos_sin_8011": d["test_balanceado"]["macro_f1_sin_8011"] >= b["test_balanceado"]["macro_f1_sin_8011"] - 0.02}
out["regla"]["cumple"] = all(out["regla"].values())
print(out["regla"])

# Oficios con familia propia según el Caso 10: ¿a qué familia los manda cada modelo? (test, texto original)
oficios = {"abogad": "8012", "contador": "8411", "ingenier": "8110", "medic": "8512",
           "enfermer": "8510", "docente": "8610", "conductor": "7811"}
out["oficios"] = []
for kw, fam10 in oficios.items():
    s = te[te.txt_norm.str.contains(kw)]
    if len(s) < 20:
        continue
    f = {"oficio": kw, "n_test": int(len(s)), "familia_caso10": fam10, "SECOP_en_8011": round(float((s.fam == "8011").mean()), 3),
         "SECOP_en_caso10": round(float((s.fam == fam10).mean()), 3)}
    for n in ("B", "D"):
        p = modelos[n].predict(s.txt_lem.tolist())
        f[f"{n}_pred_8011"], f[f"{n}_pred_caso10"] = round(float((p == "8011").mean()), 3), round(float((p == fam10).mean()), 3)
    out["oficios"].append(f)
print(pd.DataFrame(out["oficios"]).to_string(index=False))
if len(sys.argv) <= 1:
    (REPORTS / "destilacion_8011.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    guardar(modelos["D"], MODELO_APP)
