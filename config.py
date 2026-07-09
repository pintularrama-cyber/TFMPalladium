# ==========================================
# CONFIGURACIÓN CENTRALIZADA - PALLADIUM TFM
# ==========================================

import os

# ══════════════════════════════════════════
# ▶  RUTAS — CAMBIA AQUÍ SEGÚN TU MÁQUINA
# ══════════════════════════════════════════
#
# OPCIÓN A: Usa el directorio del propio proyecto (por defecto)
#   BASE_DIR se calcula automáticamente. Pon los .joblib y CSVs
#   en la misma carpeta que app.py y listo.
#
# OPCIÓN B: Rutas manuales para Mac/Windows
#   Comenta BASE_DIR y descomenta las líneas de tu sistema:

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_MODELOS      = os.path.join(BASE_DIR, 'modelos')       # pipeline_*.joblib
RUTA_DATOS        = os.path.join(BASE_DIR, 'datos')         # Reservas_22_23.csv
RUTA_ENTRENAMIENTO = os.path.join(BASE_DIR, 'entrenamiento') # df_*.csv (solo para stats)
# ══════════════════════════════════════════════════════════════

# ==========================================
# CONFIGURACIÓN FLASK
# ==========================================

FLASK_HOST = 'localhost'
FLASK_PORT = 5050
DEBUG = True
SECRET_KEY = 'palladium_tfm_2024_secretkey_abc123'

# ==========================================
# USUARIOS (usuario: contraseña)
# ==========================================

USUARIOS = {
    'diego': 'diego123',
    'erik': 'erik123',
    'guillermo': 'guillermo123',
    'raul': 'raul123'
}

# ==========================================
# CONFIGURACIÓN DE SEGMENTOS
# ==========================================

UMBRALES_RIESGO = {
    'PEQUEÑO': {'alto': 38, 'medio': 19},
    'MEDIANO': {'alto': 57, 'medio': 28},
    'GRANDE':  {'alto': 45, 'medio': 22},
}

# AUC-ROC de los modelos OPTIMIZADOS (GridSearch/RandomSearch), medidos sobre el
# conjunto de TEST en el notebook "Modelos avanzados I (con tunning)"
# (carpeta drive documentos/colabs). Valores verificados, no inventados.
# El MEDIANO usa los mismos hiperparámetros que el .joblib de producción
# (n_estimators=280, max_depth=11, lr=0.08). Datos: 27.113 / 152.768 / 965.010 filas.
SEGMENTOS = {
    'PEQUEÑO': {
        'min_habs': 0,
        'max_habs': 250,
        'modelo_archivo': 'pipeline_pequeno.joblib',
        'algoritmo': 'Random Forest',
        'auc_roc': 0.8473
    },
    'MEDIANO': {
        'min_habs': 250,
        'max_habs': 500,
        'modelo_archivo': 'pipeline_mediano.joblib',
        'algoritmo': 'XGBoost',
        'auc_roc': 0.8668
    },
    'GRANDE': {
        'min_habs': 500,
        'max_habs': 10000,
        'modelo_archivo': 'pipeline_grande.joblib',
        'algoritmo': 'XGBoost',
        'auc_roc': 0.8513
    }
}

# ==========================================
# COLUMNAS EXACTAS DEL PIPELINE (20 features post-encoding)
# Orden idéntico al ColumnTransformer entrenado
# ==========================================

COLUMNAS_PIPELINE = [
    'ANTELACION_DIAS', 'ADR', 'NOCHES', 'NENES', 'BEBES',
    'SEGMENTO_COD', 'PAIS_AGRUPADO_COD', 'HABITACION_LIMPIA_COD', 'GRUPO_TIPO_COD',
    'FUENTE_NEGOCIO_DIRECT SALES', 'FUENTE_NEGOCIO_E-COMMERCE',
    'FUENTE_NEGOCIO_OTHERS', 'FUENTE_NEGOCIO_T.O. / T.A.',
    'PAX_TIPO_PAREJAS', 'PAX_TIPO_SINGLE',
    'TEMPORADA_BAJA', 'TEMPORADA_MEDIA',
    'DISTANCIA_LARGO', 'DISTANCIA_MEDIO', 'DISTANCIA_NO INFO'
]

# Media global de cancelación por segmento (del TargetEncoder entrenado).
# Se usa como prior para SEGMENTO_COD, PAIS_AGRUPADO_COD, HABITACION_LIMPIA_COD
# cuando no se dispone del mapeo original string→float.
GLOBAL_MEANS = {
    'PEQUEÑO': 0.48727524204702627,
    'MEDIANO': 0.28362544389349830,
    'GRANDE':  0.39555678179500730,
}

# ==========================================
# VALORES PARA DROPDOWNS
# ==========================================

# Categorías reales de habitación (de documentacion/CAT_HABITACIONES.csv)
HABITACIONES_LIMPIAS = [
    'Suite', 'Junior Suite', 'Deluxe', 'Superior', 'Presidential', 'Standard'
]

PAISES_AGRUPADOS = [
    'Alemania', 'Argentina', 'Belgica', 'Brasil', 'Canada', 'Colombia',
    'Dinamarca', 'España', 'Francia', 'Holanda', 'Italia', 'Mexico',
    'Noruega', 'Portugal', 'Reino_Unido', 'Rusia', 'Suecia', 'USA', 'Otros'
]

# Proxy geográfico: país → distancia para la feature DISTANCIA_*
DISTANCIA_POR_PAIS = {
    'España': 'Corto', 'Portugal': 'Corto', 'Francia': 'Corto',
    'Belgica': 'Corto', 'Holanda': 'Corto', 'Italia': 'Corto',
    'Alemania': 'Medio', 'Suecia': 'Medio', 'Noruega': 'Medio',
    'Dinamarca': 'Medio', 'Reino_Unido': 'Medio', 'Rusia': 'Medio',
    'USA': 'Largo', 'Canada': 'Largo', 'Mexico': 'Largo',
    'Argentina': 'Largo', 'Colombia': 'Largo', 'Brasil': 'Largo',
    'Otros': 'No Info',
}

# ==========================================
# TRANSFORMACIÓN DESDE RESERVAS_22_23.CSV
# (extraído del notebook 00_Limpieza-2.ipynb)
# ==========================================

# Segmento por ID_HOTEL — 8 hoteles reales (de documentacion/catalogo_hoteles_final.csv)
HOTEL_SIZE_MAPPING = {
    99: 'PEQUEÑO',
    6: 'MEDIANO', 9: 'MEDIANO',
    83: 'GRANDE', 92: 'GRANDE', 96: 'GRANDE', 106: 'GRANDE', 107: 'GRANDE',
}

# Nombres reales de los hoteles (de documentacion/catalogo_hoteles_final.csv)
HOTEL_NOMBRES = {
    6:   'Palladium Vallarta',
    9:   'Dominican Fiesta Hotel & Casino',
    83:  'Grand Palladium Imbassai Resort & Spa',
    92:  'Grand Palladium Jamaica & Lady Hamilton Resort',
    96:  'Complejo Costa Mujeres',
    99:  'TRS Cap Cana',
    106: 'Complejo Riviera Maya',
    107: 'Complejo Punta Cana',
}

# Status → STATUS_BOOL (Cell 56)
MAPA_STATUS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 9: 1}

# Temporada por mes de llegada (Cell 75)
TEMPORADA_MESES = {
    12: 'ALTA', 1: 'ALTA', 2: 'ALTA', 3: 'ALTA', 4: 'ALTA', 5: 'ALTA',
    6: 'MEDIA', 10: 'MEDIA', 11: 'MEDIA',
    7: 'BAJA', 8: 'BAJA', 9: 'BAJA',
}

# País original → PAIS_AGRUPADO (Cell 112 + config)
MAPA_PAIS_AGRUPADO = {
    'ESTADOS UNIDOS': 'USA', 'USA': 'USA',
    'MEXICO': 'Mexico', 'MÉXICO': 'Mexico',
    'CANADA': 'Canada', 'CANADÁ': 'Canada',
    'BRASIL': 'Brasil', 'BRAZIL': 'Brasil',
    'ARGENTINA': 'Argentina',
    'ESPAÑA': 'España', 'ESPANA': 'España',
    'REINO UNIDO': 'Reino_Unido', 'UK': 'Reino_Unido',
    'ALEMANIA': 'Alemania', 'GERMANY': 'Alemania',
    'COLOMBIA': 'Colombia',
    'FRANCIA': 'Francia', 'FRANCE': 'Francia',
    'BELGICA': 'Belgica', 'BÉLGICA': 'Belgica',
    'HOLANDA': 'Holanda', 'PAÍSES BAJOS': 'Holanda',
    'ITALIA': 'Italia',
    'PORTUGAL': 'Portugal',
    'RUSIA': 'Rusia',
    'SUECIA': 'Suecia',
    'NORUEGA': 'Noruega',
    'DINAMARCA': 'Dinamarca',
}

# Tipo de cambio a USD (Cell 83-84)
TIPO_CAMBIO_USD = {
    'PESO DOMINICANO': 55.0,
    'PESO MEXICANO': 19.0,
    'REAL': 5.05,
    'DOLAR JAMAICANO': 153.8,
}

# ==========================================
# COLORES
# ==========================================

COLORES = {
    'alto_riesgo': '#e74c3c',
    'medio_riesgo': '#f39c12',
    'bajo_riesgo': '#27ae60',
    'primario': '#2c3e50',
    'secundario': '#3498db',
}
