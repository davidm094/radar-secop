"""Prueba: ¿sirve entrenar con las descripciones estilo pyme generadas por Claude (data/llm/aumentado.jsonl)?

Los textos de aumento no dependen de la compuerta del re-etiquetado (su familia se conoce por construcción),
así que se prueban aparte: mismo modelo y misma partición que scripts/03, con y sin esos textos.

Se compara en tres frentes:
  1. Acuerdo con SECOP en el test natural y el balanceado (el test nunca lleva datos del LLM).
  2. Gold set, contra el consenso de las dos anotaciones de Claude (ver scripts/13).
  3. Consultas de pyme escritas por personas (data/consultas): de los procesos que una persona juzgó
     relevantes y que traen código, ¿en cuántos la familia declarada está en el top-3 del modelo para la consulta?
     Es un indicio, no una prueba: usa el código que declaran las entidades y el pool se armó con el modelo B.
Salida: reports/prueba_textos_pyme.json. No cambia el modelo de la app.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from radar.config import DATA_PROC, DATA_RAW, REPORTS, ROOT
from radar.evaluate import metricas
from radar.labels import particion_por_texto, clases_validas, balancear, excluir_gold
from radar.model import construir
from radar.text import lematizar, limpiar, preparar

df = pd.read_parquet(DATA_PROC / "contratos_prep.parquet")
clases = clases_validas(df, "fam")
df = df[df.fam.isin(clases)]
tr, te = particion_por_texto(df, "txt_norm")
tr = excluir_gold(tr, df, ROOT / "data" / "gold" / "gold_ids.csv")
te_bal = balancear(te, "fam", tope=300)

g = pd.DataFrame([json.loads(l) for l in (ROOT / "data" / "llm" / "aumentado.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])
g = g[g.fam.isin(clases)]
g["txt_lem"] = lematizar(g.texto.map(limpiar).tolist())
base = balancear(tr, "fam")[["txt_lem", "fam"]]
variantes = {"B (modelo de la app)": base, "B + textos estilo pyme": pd.concat([base, g[["txt_lem", "fam"]]])}

# Consultas de pyme y juicios humanos de relevancia
J = pd.read_excel(ROOT / "data" / "consultas" / "juicios.xlsx", dtype=str)
J["relevante"] = pd.to_numeric(J.relevante, errors="coerce").fillna(0).astype(int)
pr = pd.read_parquet(DATA_RAW / "procesos.parquet", columns=["id_del_proceso", "codigo_principal_de_categoria"]).drop_duplicates("id_del_proceso")
pr["fam"] = pr.codigo_principal_de_categoria.str.extract(r"V1\.(\d{4})")[0]
J = J.merge(pr[["id_del_proceso", "fam"]], on="id_del_proceso", how="left")
Q = J.drop_duplicates("consulta_id")[["consulta_id", "consulta"]]
R = J[(J.relevante == 1) & J.fam.notna()]

# Gold set: consenso de las dos anotaciones de Claude
gold = pd.read_csv(ROOT / "data" / "gold" / "gold_anotaciones.csv", dtype=str).fillna("")
cons = gold[(gold.claude_sonnet == gold.claude_fable) & ~gold.claude_sonnet.isin(["NC", ""])]

out = {"textos_pyme": int(len(g)), "familias": int(g.fam.nunique()), "consultas_con_relevantes": int(R.consulta_id.nunique()),
       "procesos_relevantes_con_codigo": int(len(R)), "gold_consenso_n": int(len(cons)), "variantes": {}}
for nombre, d in variantes.items():
    m = construir().fit(d.txt_lem, d.fam)
    r = {"n_entrenamiento": int(len(d))}
    for conj, t in [("test_natural", te), ("test_balanceado", te_bal)]:
        f = metricas(m, t.txt_lem, t.fam, conj)
        r[conj] = {k: round(float(f[k]), 3) for k in ("accuracy", "macro_f1", "top3")}
    P = m.predict_proba([preparar(x) for x in cons.objeto_del_contrato])
    top = m.classes_[np.argsort(-P, 1)[:, :3]]
    r["gold_consenso_claude"] = {"accuracy": round(float((top[:, 0] == cons.claude_sonnet.values).mean()), 3),
                                 "top3": round(float(np.mean([a in b for a, b in zip(cons.claude_sonnet.values, top)])), 3)}
    Pq = m.predict_proba([preparar(x) for x in Q.consulta])
    tq = dict(zip(Q.consulta_id, m.classes_[np.argsort(-Pq, 1)[:, :3]]))
    en3 = np.array([f in tq[q] for q, f in zip(R.consulta_id, R.fam)])
    r["consultas_pyme"] = {"familia_del_proceso_relevante_en_top3": round(float(en3.mean()), 3),
                           "promedio_por_consulta": round(float(R.assign(ok=en3).groupby("consulta_id").ok.mean().mean()), 3),
                           "confianza_media_top1": round(float(Pq.max(1).mean()), 3)}
    out["variantes"][nombre] = r
    print(nombre, json.dumps(r, ensure_ascii=False))

(REPORTS / "prueba_textos_pyme.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
