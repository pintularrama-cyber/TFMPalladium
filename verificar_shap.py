"""
Script de verificación rápida de SHAP.
Ejecuta:  python verificar_shap.py
Te dice si SHAP funciona con tus modelos, sin arrancar la web.
Puedes borrar este archivo cuando quieras.
"""
import sys, os
import joblib
import numpy as np

# Inyectar las clases del pipeline en __main__ (igual que hace app.py al cargar los .joblib)
import preprocesamiento as _prep
_main = sys.modules.get('__main__')
for _name in dir(_prep):
    if not _name.startswith('_'):
        setattr(_main, _name, getattr(_prep, _name))

from config import COLUMNAS_PIPELINE, SEGMENTOS, RUTA_MODELOS

print("=" * 58)
print("  VERIFICACIÓN DE SHAP — Palladium TFM")
print("=" * 58)

# 1. ¿SHAP se puede importar?
try:
    import shap
    print(f"[1/5] SHAP instalado ...................... OK (v{shap.__version__})")
except Exception as e:
    print(f"[1/5] SHAP NO disponible .................. FALLO: {e}")
    print("\n>>> Instala SHAP con:  pip install shap")
    sys.exit(1)

# 2. Cargar un modelo
seg = 'PEQUEÑO'
try:
    ruta = os.path.join(RUTA_MODELOS, SEGMENTOS[seg]['modelo_archivo'])
    modelo = joblib.load(ruta)
    clf = modelo.named_steps['modelo']
    print(f"[2/5] Modelo '{seg}' cargado ........... OK ({type(clf).__name__})")
except Exception as e:
    print(f"[2/5] No se pudo cargar el modelo ........ FALLO: {e}")
    sys.exit(1)

# 3. Crear el explainer SHAP
try:
    explainer = shap.TreeExplainer(clf)
    print("[3/5] TreeExplainer creado ............... OK")
except Exception as e:
    print(f"[3/5] No se pudo crear el explainer ...... FALLO: {e}")
    sys.exit(1)

# 4. Predicción de ejemplo
ejemplo = {c: 0.0 for c in COLUMNAS_PIPELINE}
ejemplo.update({
    'ANTELACION_DIAS': 120, 'ADR': 250, 'NOCHES': 5,
    'FUENTE_NEGOCIO_E-COMMERCE': 1, 'PAX_TIPO_PAREJAS': 1,
    'SEGMENTO_COD': 0.48, 'PAIS_AGRUPADO_COD': 0.48, 'HABITACION_LIMPIA_COD': 0.48,
})
X = np.array([[float(ejemplo[c]) for c in COLUMNAS_PIPELINE]], dtype=np.float64)
try:
    prob = float(clf.predict_proba(X)[0][1]) * 100
    print(f"[4/5] Predicción de ejemplo ............. OK ({prob:.1f}% cancelación)")
except Exception as e:
    print(f"[4/5] Fallo al predecir .................. FALLO: {e}")
    sys.exit(1)

# 5. Calcular valores SHAP
try:
    sv = explainer.shap_values(X)
    arr = np.array(sv)
    if arr.ndim == 3:
        vals = arr[0, :, 1] if arr.shape[2] > 1 else arr[0, :, 0]
    elif arr.ndim == 2:
        vals = arr[0]
    else:
        vals = arr.ravel()[:len(COLUMNAS_PIPELINE)]
    vals = np.asarray(vals, dtype=np.float64).ravel()[:len(COLUMNAS_PIPELINE)]
    print("[5/5] Valores SHAP calculados ........... OK")
except Exception as e:
    print(f"[5/5] Fallo al calcular SHAP ............. FALLO: {e}")
    sys.exit(1)

# Mostrar los factores más influyentes
orden = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)[:5]
print("\n  TOP 5 factores SHAP de esta reserva de ejemplo:")
print("  " + "-" * 52)
for i in orden:
    signo = "sube el riesgo" if vals[i] > 0 else "baja el riesgo"
    flecha = "^" if vals[i] > 0 else "v"
    print(f"   {flecha} {COLUMNAS_PIPELINE[i]:30} {vals[i]:+.4f}  ({signo})")

print("\n" + "=" * 58)
print("  RESULTADO: SHAP FUNCIONA CORRECTAMENTE")
print("  Las explicaciones de la app saldran del modelo (SHAP).")
print("=" * 58)
