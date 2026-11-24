"""Deja la demo lista con un solo comando: corre solo los pasos que falten y dice cómo abrir la app.

    python scripts/preparar_demo.py

Usa los datos congelados (paso 00), entrena el modelo de la app (paso 15) y construye el índice de procesos
(paso 07, ~20 min). Si ya existen, no los repite. Al final comprueba que la app tiene todo lo que necesita.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from radar.config import DATA_PROC, DATA_RAW, MODELS, REPORTS, MODELO_APP

PASOS = [
    ("00_usar_datos_congelados.py", [DATA_PROC / "contratos_prep.parquet", DATA_RAW / "unspsc_v14.parquet"], "datos congelados"),
    ("15_destilacion_8011.py", [MODELS / f"{MODELO_APP}.joblib"], "modelo de la app (unos minutos)"),
    ("07_indexar_procesos.py", [DATA_PROC / "procesos_idx.parquet", DATA_PROC / "procesos_emb.npy"], "índice de procesos (unos 20 minutos)"),
]
for script, salidas, que in PASOS:
    if all(p.exists() for p in salidas):
        print(f"✓ {que}: ya está")
        continue
    print(f"→ {que}: corriendo scripts/{script} …", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT)

faltan = [p for p in [REPORTS / "discrepancias.csv", REPORTS / "destilacion_8011.json"] if not p.exists()]
if faltan:
    sys.exit(f"Faltan reportes que la app usa: {[p.name for p in faltan]}. Vienen en el repo; revise el clon.")
print("\nTodo listo. Abra la app con:\n\n    streamlit run app/app.py\n")
