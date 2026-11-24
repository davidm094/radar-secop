"""Pruebas básicas del repo. Corren sin entrenar nada: `python tests/test_basico.py` o `pytest tests/`.

Comprueban tres cosas de las que dependen las cifras del documento:
  1. la limpieza de texto quita las fórmulas de contratación y deja el oficio;
  2. el gold set tiene 300 contratos y ninguno está en el split de entrenamiento congelado;
  3. los reportes publicados son coherentes entre sí.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd


def test_limpieza_quita_formulas():
    from radar.text import preparar
    t = preparar("Prestación de servicios profesionales como abogado para el acompañamiento jurídico de la personería municipal")
    assert "abogado" in t and "juridico" in t
    assert "prestacion" not in t and "profesionales" not in t


def test_gold_fuera_del_entrenamiento():
    gold = pd.read_csv(ROOT / "data" / "gold" / "gold_anotaciones.csv", dtype=str).fillna("")
    assert len(gold) == 300 and gold.item_id.is_unique
    assert (gold.humano != "").sum() == 215
    split = pd.read_parquet(ROOT / "data" / "beto" / "split_familia.parquet")
    col_id = next(c for c in split.columns if c.startswith("id"))
    train = split[split["split"] == "train"] if "split" in split.columns else split
    assert not set(train[col_id].astype(str)) & set(gold.id_contrato.astype(str))


def test_reportes_coherentes():
    ref = json.loads((ROOT / "reports" / "gold_referencias.json").read_text(encoding="utf-8"))
    assert ref["n"] == 300 and ref["n_humano"] == 215
    c = ref["contra_referencia"]
    assert c["H_humano"]["modelo_app"]["accuracy"] < c["C_consenso_claude"]["modelo_app"]["accuracy"]
    for r in c.values():   # el hallazgo firme: 8011 mal declarado con cualquier referencia
        assert r["error_secop"]["por_grupo"]["8011"]["error"] > 0.9
    m = pd.read_csv(ROOT / "reports" / "metricas_familia.csv")
    assert len(m) >= 4
    d = pd.read_csv(ROOT / "reports" / "discrepancias.csv")
    assert len(d) == 1640 and (d.conf >= 0.70).all()


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
            print("ok", nombre)
