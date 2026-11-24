"""Revisión humana de las discrepancias que marca la app (¿la alerta acierta?).

Paso 1:  python scripts/11_revisar_discrepancias.py muestra
  toma 100 contratos AL AZAR (semilla 42) de reports/discrepancias.csv (confianza >= 0,70) y crea
  reports/discrepancias_revision.xlsx con la columna 'revision' vacía. Al azar y no los 100 de mayor
  confianza, para que la cifra represente a todo lo que la app marca.
Paso 2:  llenar 'revision' con: modelo | secop | ninguna | no_clasificable
Paso 3:  python scripts/11_revisar_discrepancias.py evaluar   -> reports/discrepancias_revision_resumen.json

La muestra se sacó de las alertas del modelo con etiquetas de SECOP (familia_v1). Cuando la app pasó al modelo
con etiquetas de 8011 corregidas (familia_v2), no se repitió la revisión:
         python scripts/11_revisar_discrepancias.py trasladar
  cuenta cuántos de los 100 contratos revisados siguen siendo alerta del modelo de la app con la misma familia
  sugerida, y qué veredicto tenían. Las alertas nuevas del modelo de la app quedan sin revisión humana.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from radar.config import REPORTS, SEED

XLSX = REPORTS / "discrepancias_revision.xlsx"
VALIDOS = {"modelo", "secop", "ninguna", "no_clasificable"}


def muestra(n: int = 100):
    d = pd.read_csv(REPORTS / "discrepancias.csv", dtype={"fam": str, "pred": str})
    m = d.sample(min(n, len(d)), random_state=SEED).sort_values("id_contrato")
    cols = ["id_contrato", "nombre_entidad", "objeto_del_contrato", "fam", "nombre_declarado", "pred",
            "nombre_predicho"]   # sin la confianza: no sesgar al revisor
    m[cols].assign(revision="", comentario="").to_excel(XLSX, index=False)
    print(f"{len(m)} contratos de {len(d)} -> {XLSX}")


def evaluar():
    r = pd.read_excel(XLSX, dtype=str)
    r["revision"] = r.revision.fillna("").str.strip().str.lower()
    malos = r[~r.revision.isin(VALIDOS)]
    if len(malos):
        sys.exit(f"{len(malos)} filas con 'revision' vacía o inválida (use {sorted(VALIDOS)}): "
                 f"{malos.id_contrato.head(5).tolist()}")
    p = r.revision.value_counts(normalize=True).round(3).to_dict()
    out = {"n": len(r), "proporciones": p, "precision_alerta": p.get("modelo", 0.0),
           "nota": "precision_alerta = fracción en que la familia sugerida por el modelo era la correcta"}
    (REPORTS / "discrepancias_revision_resumen.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


def trasladar():
    r = pd.read_excel(XLSX, dtype=str)
    r["revision"] = r.revision.fillna("").str.strip().str.lower()
    d = pd.read_csv(REPORTS / "discrepancias.csv", dtype={"fam": str, "pred": str}).set_index("id_contrato")
    r["pred_app"] = r.id_contrato.map(d.pred)
    misma = r[r.pred_app == r.pred]
    ruta = REPORTS / "discrepancias_revision_resumen.json"
    out = json.loads(ruta.read_text(encoding="utf-8"))
    ids_b = set(pd.read_csv(REPORTS / "discrepancias_modelo_secop.csv", usecols=["id_contrato"]).id_contrato)
    out["modelo_revisado"] = "familia_v1 (etiquetas de SECOP)"
    out["traslado_al_modelo_de_la_app"] = {
        "alertas_del_modelo_de_la_app": int(len(d)),
        "ya_eran_alerta_del_modelo_revisado": int(d.index.isin(ids_b).sum()),
        "alertas_nuevas_sin_revision_humana": int((~d.index.isin(ids_b)).sum()),
        "revisadas_que_siguen_como_alerta": int(r.pred_app.notna().sum()),
        "revisadas_que_siguen_con_la_misma_familia": int(len(misma)),
        "veredicto_de_esas": misma.revision.value_counts().to_dict(),
        "precision_en_esas": round(float((misma.revision == "modelo").mean()), 3),
        "veredicto_de_las_que_ya_no_son_alerta": r[r.pred_app != r.pred].revision.value_counts().to_dict(),
        "nota": "la revisión humana solo cubre las alertas que el modelo de la app comparte con el modelo revisado"}
    ruta.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out["traslado_al_modelo_de_la_app"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"muestra": muestra, "evaluar": evaluar, "trasladar": trasladar}[sys.argv[1]]()
