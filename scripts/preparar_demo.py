"""Deja la demo lista con un solo comando: corre solo los pasos que falten y dice cómo abrir la app.

    python scripts/preparar_demo.py

Usa los datos congelados (paso 00), entrena el modelo (paso 03, ~5 min) y construye el índice de procesos
(paso 07, ~20 min). Si ya existen, no los repite. Al final comprueba que la app tiene todo lo que necesita.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from radar.config import DATA_PROC, DATA_RAW, MODELS, REPORTS

PASOS = [
    ("00_usar_datos_congelados.py", [DATA_PROC / "contratos_prep.parquet", DATA_RAW / "unspsc_v14.parquet"], "datos congelados"),
    ("03_entrenar_evaluar.py", [MODELS / "familia_v1.joblib"], "modelo (unos 5 minutos)"),
    ("07_indexar_procesos.py", [DATA_PROC / "procesos_idx.parquet", DATA_PROC / "procesos_emb.npy"], "índice de procesos (unos 20 minutos)"),
]
for script, salidas, que in PASOS:
    if all(p.exists() for p in salidas):
        print(f"✓ {que}: ya está")
        continue
    print(f"→ {que}: corriendo scripts/{script} …", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT)

faltan = [p for p in [REPORTS / "discrepancias.csv", REPORTS / "metricas_familia.csv", REPORTS / "explicabilidad_global.csv"] if not p.exists()]
if faltan:
    sys.exit(f"Faltan reportes que la app usa: {[p.name for p in faltan]}. Vienen en el repo; revise el clon.")
print("\nTodo listo. Abra la app con:\n\n    streamlit run app/app.py\n")
