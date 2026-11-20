"""Evaluación final contra el gold set, con dos referencias y su confiabilidad.

Entrada: data/gold/gold_anotaciones.csv (una fila por contrato del gold set, 300):
  humano            familia anotada por una persona (215 contratos; vacío en los otros 85)
  claude_sonnet     anotación de Claude Sonnet 5.5 (llm/reetiquetar.py, 14-oct-2026), sin ver el código de SECOP
  claude_fable      segunda anotación de Claude (Fable 5.2, 20-nov-2026), independiente de la primera y
                    sin ver el código de SECOP ni las otras etiquetas
  secop             familia declarada en SECOP (la "clave")
  estrato           estrato del muestreo (8011 por oficio / salud / otras familias)

Dos referencias, porque ninguna es perfecta (ver el informe de confiabilidad que imprime este script):
  H = etiqueta humana (215 contratos).
  C = consenso de Claude: contratos donde las dos anotaciones independientes de Claude dan la misma familia.
      No es una referencia humana, y dos modelos de la misma familia comparten sesgos.
Mide, contra cada referencia: acuerdo entre anotadores (kappa de Cohen, familia y segmento), error de SECOP
por estrato, modelo de la app (accuracy, macro-F1, top-3) y plausibilidad de la explicación (solo con las
palabras clave humanas).
Salida: reports/gold_referencias.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from radar.config import REPORTS, ROOT, SEED
from radar.explain import local
from radar.model import cargar
from radar.text import preparar

G = ROOT / "data" / "gold"


def acuerdo(a: pd.Series, b: pd.Series) -> dict:
    m = (a != "") & (b != "")
    a, b = a[m], b[m]
    return {"n": int(m.sum()),
            "familia": {"acuerdo": round(float((a == b).mean()), 3), "kappa": round(float(cohen_kappa_score(a, b)), 3)},
            "segmento": {"acuerdo": round(float((a.str[:2] == b.str[:2]).mean()), 3),
                         "kappa": round(float(cohen_kappa_score(a.str[:2], b.str[:2])), 3)}}


def grupo_estrato(e: str) -> str:
    return "8011" if e.startswith("8011") else ("salud" if e.startswith("salud") else "otras")


def main():
    d = pd.read_csv(G / "gold_anotaciones.csv", dtype=str).fillna("").set_index("item_id")
    out = {"n": len(d), "n_humano": int((d.humano != "").sum())}

    # 1. Confiabilidad: acuerdo entre cada par de fuentes
    fuentes = ["humano", "claude_sonnet", "claude_fable", "secop"]
    out["acuerdo_entre_fuentes"] = {f"{a}__{b}": acuerdo(d[a], d[b]) for a, b in combinations(fuentes, 2)}
    h = d[d.humano != ""]
    out["acuerdo_entre_fuentes"]["claude_sonnet__claude_fable__solo_215_humanos"] = acuerdo(h.claude_sonnet, h.claude_fable)
    ctrl = d[d.control_ciego == "sí"]
    out["control_ciego_60"] = {"humano__claude_fable": acuerdo(ctrl.humano, ctrl.claude_fable),
                               "humano__claude_sonnet": acuerdo(ctrl.humano, ctrl.claude_sonnet),
                               "claude_sonnet__claude_fable": acuerdo(ctrl.claude_sonnet, ctrl.claude_fable)}
    out["no_clasificable"] = {f: int((d[f] == "NC").sum()) for f in ["humano", "claude_sonnet", "claude_fable"]}

    # 2. Referencias
    cons = (d.claude_sonnet == d.claude_fable) & (d.claude_sonnet != "NC") & (d.claude_sonnet != "")
    refs = {"H_humano": d.humano.where(d.humano != "NC", ""),
            "C_consenso_claude": d.claude_sonnet.where(cons, ""),
            "HC_humano_y_consenso_coinciden": d.humano.where(cons & (d.humano == d.claude_sonnet), "")}
    d["grupo"] = d.estrato.map(grupo_estrato)

    mod = cargar("familia_v1")
    textos = pd.Series([preparar(t) for t in d.objeto_del_contrato], index=d.index)
    P = mod.predict_proba(textos.tolist())
    top = pd.DataFrame(mod.classes_[np.argsort(-P, 1)[:, :3]], index=d.index)
    d["modelo"] = top[0]
    d["modelo_conf"] = P.max(1)

    out["contra_referencia"] = {}
    for nombre, ref in refs.items():
        m = ref != ""
        r, s, p, t = ref[m], d.secop[m], d.modelo[m], top[m]
        ok_secop = s == r
        en3 = np.array([g in fila for g, fila in zip(r, t.values)])
        out["contra_referencia"][nombre] = {
            "n": int(m.sum()),
            "error_secop": {"global": round(float(1 - ok_secop.mean()), 3),
                            "por_grupo": {g: {"error": round(float(1 - ok_secop[d.grupo[m] == g].mean()), 3),
                                              "n": int((d.grupo[m] == g).sum())} for g in ["8011", "salud", "otras"]},
                            "error_a_nivel_segmento": round(float(1 - (s.str[:2] == r.str[:2]).mean()), 3)},
            "modelo_app": {"accuracy": round(float((p == r).mean()), 3),
                           "macro_f1": round(float(f1_score(r, p, average="macro", zero_division=0)), 3),
                           "top3": round(float(en3.mean()), 3),
                           "accuracy_segmento": round(float((p.str[:2] == r.str[:2]).mean()), 3),
                           "acuerdo_del_modelo_con_secop_mismos_items": round(float((p == s).mean()), 3),
                           "referencia_fuera_de_las_175_familias": round(float(1 - r.isin(mod.classes_).mean()), 3),
                           "por_grupo": {g: {"accuracy": round(float((p == r)[d.grupo[m] == g].mean()), 3),
                                             "top3": round(float(en3[(d.grupo[m] == g).values].mean()), 3)}
                                         for g in ["8011", "salud", "otras"]}},
        }
        if nombre != "H_humano":   # ¿a dónde deberían ir los contratos declarados en 8011?
            tr = r[(s == "8011") & (r != "8011")].value_counts().head(8)
            out["contra_referencia"][nombre]["destino_de_los_8011_mal_declarados"] = tr.to_dict()

    # 3. Alertas de discrepancia (confianza >= 0,70 y familia distinta de SECOP) dentro del gold set
    al = (d.modelo_conf >= 0.70) & (d.modelo != d.secop)
    out["alertas_en_gold"] = {"n": int(al.sum())}
    for nombre, ref in refs.items():
        m = al & (ref != "")
        if m.sum():
            out["alertas_en_gold"][nombre] = {"n": int(m.sum()), "modelo_correcto": round(float((d.modelo[m] == ref[m]).mean()), 3),
                                              "secop_correcto": round(float((d.secop[m] == ref[m]).mean()), 3)}

    # 4. Plausibilidad: términos que resalta el modelo vs. palabras clave humanas (token-F1, ERASER)
    rng = np.random.RandomState(SEED)
    f1s, azar = [], []
    for t, pk, fam in zip(textos[h.index], h.palabras_clave_humano, d.modelo[h.index]):
        humano = set().union(*[set(preparar(x).split()) for x in pk.split(";") if x.strip()]) if pk.strip(";") else set()
        if not humano or not t:
            continue
        exp = local(mod, t, fam, k=10)
        resaltado = set(" ".join(exp[exp.contribucion > 0].head(5).termino).split())
        palabras = t.split()
        al_azar = set(rng.choice(palabras, size=min(len(resaltado) or 1, len(palabras)), replace=False))
        for conj, lista in ((resaltado, f1s), (al_azar, azar)):
            inter = len(conj & humano)
            lista.append(0.0 if inter == 0 else 2 * inter / (len(conj) + len(humano)))
    out["plausibilidad_token_f1"] = {"modelo": round(float(np.mean(f1s)), 3), "azar": round(float(np.mean(azar)), 3), "n": len(f1s)}

    d[["humano", "claude_sonnet", "claude_fable", "secop", "modelo", "modelo_conf", "grupo"]].assign(
        consenso_claude=refs["C_consenso_claude"]).to_csv(REPORTS / "gold_detalle.csv")
    (REPORTS / "gold_referencias.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
