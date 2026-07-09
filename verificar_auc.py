"""
Calcula el AUC-ROC REAL de cada modelo sobre tus datos.
Ejecuta:  python verificar_auc.py
Te da el AUC verdadero para poner en config.py (en vez de un número sin respaldo).
Puedes borrar este archivo cuando quieras.
"""
import sys, os
import joblib
import numpy as np

# Inyectar clases del pipeline (igual que app.py)
import preprocesamiento as _prep
_main = sys.modules.get('__main__')
for _name in dir(_prep):
    if not _name.startswith('_'):
        setattr(_main, _name, getattr(_prep, _name))

import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from config import COLUMNAS_PIPELINE, SEGMENTOS, RUTA_MODELOS, RUTA_ENTRENAMIENTO

ARCHIVOS = {
    'PEQUEÑO': 'df_pequeno.csv',
    'MEDIANO': 'df_mediano.csv',
    'GRANDE':  'df_grande.csv',
}

print("=" * 60)
print("  AUC-ROC REAL DE LOS MODELOS — Palladium TFM")
print("=" * 60)
print("  (calculado sobre los datos de entrenamiento disponibles)")
print()

for seg, archivo in ARCHIVOS.items():
    ruta_csv = os.path.join(RUTA_ENTRENAMIENTO, archivo)
    if not os.path.exists(ruta_csv):
        print(f"  {seg:9} -> AVISO: no se encuentra {archivo} (¿es el CSV grande no incluido?)")
        continue
    try:
        modelo = joblib.load(os.path.join(RUTA_MODELOS, SEGMENTOS[seg]['modelo_archivo']))
        clf = modelo.named_steps['modelo']
        df = pd.read_csv(ruta_csv, low_memory=False)

        # Comprobar que están todas las columnas necesarias
        faltan = [c for c in COLUMNAS_PIPELINE if c not in df.columns]
        if faltan or 'STATUS_BOOL' not in df.columns:
            print(f"  {seg:9} -> AVISO: faltan columnas {faltan or ['STATUS_BOOL']}")
            continue

        X = df[COLUMNAS_PIPELINE].astype(float).values
        y = df['STATUS_BOOL'].astype(int).values

        prob = clf.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y, prob)
        acc = accuracy_score(y, pred)
        algo = SEGMENTOS[seg]['algoritmo']
        print(f"  {seg:9} | {algo:14} | AUC = {auc:.4f} | Accuracy = {acc:.4f} | n={len(y):,}")
    except Exception as e:
        print(f"  {seg:9} -> ERROR: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("  Copia estos AUC reales a config.py (campo 'auc_roc').")
print("  Nota: calculado sobre los datos disponibles; si quieres el")
print("  AUC de test puro necesitarías el notebook de entrenamiento.")
print("=" * 60)
