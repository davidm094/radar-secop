"""Paso 3 (validación): todo lo que se mide contra el gold set humano.

Entradas en data/gold/:
  gold_set_anotador_David.xlsx, gold_set_anotador_Dierick.xlsx   (hoja 'anotacion')
  gold_set_CLAVE_no_abrir.xlsx                                     (código declarado en SECOP)
  gold_resuelto.xlsx (opcional: item_id, familia_gold, no_clasificable, disputado) tras discutir desacuerdos;
    se arma desde la plantilla que crea  python scripts/08_acuerdo_gold.py --desacuerdos
y, si existe, data/llm/gold_claude.jsonl (salida de llm/reetiquetar.py sobre los mismos 300).

Mide:
  1. Acuerdo entre anotadores (kappa de Cohen) en G031–G300, a nivel de familia y de segmento.
  2. Tasa de error de la etiqueta de SECOP por estrato (¿qué tan sucia está 8011?).
  3. Acuerdo de Claude con el gold set (decide si el re-etiquetado a escala es confiable).
  4. Modelo vs gold: accuracy, macro-F1 y top-3, comparado con su acuerdo con SECOP en los mismos ítems.
  5. Plausibilidad de la explicación: solapamiento entre las palabras que resalta el modelo y las
     palabras_clave que marcaron los humanos (ERASER, token-F1), contra una referencia al azar.
Salidas: reports/gold_resumen.json y reports/gold_detalle.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from radar.config import ROOT, REPORTS, SEED
from radar.explain import local
from radar.model import cargar
from radar.text import preparar

G = ROOT / "data" / "gold"
SI = {"si", "sí", "x", "true", "1", "yes"}


def fam4(v) -> str:
    s = str(v).strip()
    return s[:4] if s[:4].isdigit() else ""


def leer_anotador(nombre: str) -> pd.DataFrame:
    d = pd.read_excel(G / f"gold_set_anotador_{nombre}.xlsx", sheet_name="anotacion")
    d["fam"] = d.familia_anotada.map(fam4)
    d["fam2"] = d.segunda_familia.map(fam4)
    d["nc"] = d.no_clasificable.astype(str).str.strip().str.lower().isin(SI)
    d.loc[d.nc, "fam"] = "NC"
    return d.set_index("item_id")


def kappa(a: pd.Series, b: pd.Series) -> dict:
    m = (a != "") & (b != "")
    return {"kappa": float(cohen_kappa_score(a[m], b[m])), "acuerdo_bruto": float((a[m] == b[m]).mean()),
            "n": int(m.sum())}


def tokens(texto: str) -> set:
    return set(preparar(texto).split())


def main():
    global MODELOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["familia_v1"])
    ap.add_argument("--desacuerdos", action="store_true",
                    help="solo kappa + lista de desacuerdos y plantilla de gold_resuelto.xlsx (no abre la clave ni Claude)")
    args = ap.parse_args()
    MODELOS = args.modelos
    out = {}
    A, B = leer_anotador("David"), leer_anotador("Dierick")
    vacias = {n: int((d.fam == "").sum()) for n, d in (("David", A), ("Dierick", B))}
    if any(vacias.values()):
        print(f"Aviso: filas sin familia ni 'no clasificable' por anotador: {vacias}")

    if args.desacuerdos:
        ind = A.index[~A.fase.astype(str).str.startswith("1")]
        print("kappa familia (G031-G300):", kappa(A.loc[ind, "fam"], B.loc[ind, "fam"]))
        print("kappa segmento:", kappa(A.loc[ind, "fam"].str[:2], B.loc[ind, "fam"].str[:2]))
        dis = A.fam != B.fam
        tabla = pd.DataFrame({"objeto_del_contrato": A.objeto_del_contrato, "nombre_entidad": A.nombre_entidad,
                              "David": A.familia_anotada, "David_2": A.segunda_familia, "David_comentario": A.comentario,
                              "Dierick": B.familia_anotada, "Dierick_2": B.segunda_familia,
                              "Dierick_comentario": B.comentario})
        tabla[dis].to_excel(G / "gold_desacuerdos.xlsx")
        plantilla = pd.DataFrame({"familia_gold": A.fam.where(~dis & (A.fam != "NC"), ""),
                                  "no_clasificable": np.where(~dis & (A.fam == "NC"), "sí", ""),
                                  "disputado": "", "resuelto_en_discusion": np.where(dis, "pendiente", "")})
        plantilla.index.name = "item_id"
        plantilla.to_excel(G / "gold_resuelto_plantilla.xlsx")
        print(f"{int(dis.sum())} desacuerdos -> data/gold/gold_desacuerdos.xlsx; "
              f"plantilla -> data/gold/gold_resuelto_plantilla.xlsx (guárdela como gold_resuelto.xlsx)")
        return
    clave = pd.read_excel(G / "gold_set_CLAVE_no_abrir.xlsx").set_index("item_id")
    clave["fam_secop"] = clave.familia_declarada.astype(str).str.zfill(4)

    # 1. Acuerdo entre anotadores, solo fase independiente
    ind = A.index[~A.fase.astype(str).str.startswith("1")]
    out["kappa_familia"] = kappa(A.loc[ind, "fam"], B.loc[ind, "fam"])
    out["kappa_segmento"] = kappa(A.loc[ind, "fam"].str[:2], B.loc[ind, "fam"].str[:2])
    out["no_clasificable_por_anotador"] = {"David": float(A.nc.mean()), "Dierick": float(B.nc.mean())}

    # Etiqueta gold: la resuelta en la discusión si existe; si no, solo donde ambos coinciden
    res = G / "gold_resuelto.xlsx" if (G / "gold_resuelto.xlsx").exists() else G / "gold_resuelto.csv"
    if res.exists():
        if res.suffix == ".xlsx":
            r = pd.read_excel(res, dtype=str).fillna("").set_index("item_id")
        else:  # acepta , o ; (Excel en español)
            r = pd.read_csv(res, dtype=str, sep=None, engine="python", encoding="utf-8-sig").fillna("").set_index("item_id")
        gold = r.familia_gold.map(fam4)
        gold[r.get("no_clasificable", "").astype(str).str.lower().isin(SI)] = "NC"
        gold[r.get("disputado", "").astype(str).str.lower().isin(SI)] = ""
        out["fuente_gold"] = res.name
    else:
        gold = A.fam.where(A.fam == B.fam, "")
        out["fuente_gold"] = "coincidencias entre anotadores (desacuerdos excluidos)"
    det = pd.DataFrame({"gold": gold, "david": A.fam, "dierick": B.fam,
                        "objeto": A.objeto_del_contrato, "palabras_clave": A.palabras_clave.fillna("").astype(str)
                        + ";" + B.palabras_clave.fillna("").astype(str)}).join(clave[["estrato", "fam_secop"]])
    det = det[det.gold != ""]
    out["n_gold"] = int(len(det))
    out["n_no_clasificable"] = int((det.gold == "NC").sum())
    ev = det[det.gold != "NC"].copy()

    # 2. Error de la etiqueta SECOP por estrato
    ev["secop_ok"] = ev.fam_secop == ev.gold
    out["error_secop"] = {"global": float(1 - ev.secop_ok.mean()),
                          "por_estrato": (1 - ev.groupby("estrato").secop_ok.mean()).round(3).to_dict()}
    tr = ev[~ev.secop_ok].groupby(["fam_secop", "gold"]).size().sort_values(ascending=False).head(10)
    out["error_secop"]["transiciones_top"] = {f"{a}->{b}": int(n) for (a, b), n in tr.items()}

    # 3. Claude vs gold
    cj = ROOT / "data" / "llm" / "gold_claude.jsonl"
    if cj.exists():
        c = pd.DataFrame([json.loads(l) for l in cj.read_text(encoding="utf-8").splitlines() if l.strip()])
        c = c.set_index("id")
        c["fam_c"] = np.where(c.no_clasificable, "NC", c.familia)
        det = det.join(c[["fam_c", "confianza"]].rename(columns={"confianza": "conf_claude"}))
        m = det.fam_c.notna()
        out["claude_vs_gold"] = {**kappa(det.loc[m, "gold"], det.loc[m, "fam_c"]),
                                 "por_confianza": det[m].assign(ok=det.gold == det.fam_c)
                                 .groupby("conf_claude").ok.agg(["mean", "size"]).round(3).to_dict("index"),
                                 "vs_secop_acuerdo": float((det.loc[m, "fam_c"] == det.loc[m, "fam_secop"]).mean())}
        ev = ev.join(c[["fam_c"]])

    # 4. Modelos vs gold (y vs SECOP en los mismos ítems)
    textos = [preparar(t) for t in ev.objeto]
    out["modelos_vs_gold"] = {}
    for nombre_m in MODELOS:
        mod = cargar(nombre_m)
        P = mod.predict_proba(textos)
        top = mod.classes_[np.argsort(-P, 1)[:, :3]]
        pred = top[:, 0]
        out["modelos_vs_gold"][nombre_m] = {
            "accuracy": float((pred == ev.gold).mean()),
            "macro_f1": float(f1_score(ev.gold, pred, average="macro", zero_division=0)),
            "top3": float(np.mean([g in t for g, t in zip(ev.gold, top)])),
            "accuracy_vs_secop_mismos_items": float((pred == ev.fam_secop).mean()),
            "gold_fuera_de_las_familias_del_modelo": float(1 - ev.gold.isin(mod.classes_).mean()),
            "n": int(len(ev))}
        if nombre_m == MODELOS[0]:
            mod0, ev["pred"] = mod, pred
    out["secop_vs_gold_accuracy"] = float(ev.secop_ok.mean())
    if "fam_c" in ev:
        m = ev.fam_c.notna()
        out["claude_vs_gold_accuracy_mismos_items"] = float((ev.fam_c[m] == ev.gold[m]).mean())

    # 5. Plausibilidad: palabras resaltadas por el modelo vs palabras_clave humanas
    rng = np.random.RandomState(SEED)
    f1s, f1_azar = [], []
    for t, pk, fam in zip(textos, ev.palabras_clave, ev.pred):
        humano = set().union(*[tokens(p) for p in pk.split(";") if p.strip()]) if pk.strip(";") else set()
        if not humano or not t:
            continue
        exp = local(mod0, t, fam, k=10)
        modelo = set(" ".join(exp[exp.contribucion > 0].head(5).termino).split())
        palabras = t.split()
        azar = set(rng.choice(palabras, size=min(len(modelo) or 1, len(palabras)), replace=False))
        for conj, lista in ((modelo, f1s), (azar, f1_azar)):
            inter = len(conj & humano)
            lista.append(0.0 if inter == 0 else 2 * inter / (len(conj) + len(humano)))
    out["plausibilidad_token_f1"] = {"modelo": float(np.mean(f1s)), "azar": float(np.mean(f1_azar)), "n": len(f1s)}

    det.to_csv(REPORTS / "gold_detalle.csv")
    (REPORTS / "gold_resumen.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
