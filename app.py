# ==========================================
# APLICACIÓN FLASK - PREDICTOR PALLADIUM (PRO 2GB + SHAP GLOBAL)
# ==========================================

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
from preprocesamiento import TargetEncoderSmoothing, BinaryEncoderCustom  # necesario para deserializar joblib
import pandas as pd
from config import *
import os
from datetime import datetime
from datetime import datetime as _dt

app = Flask(__name__)
app.secret_key = SECRET_KEY

# ==========================================
# MODELOS Y EXPLAINERS SHAP PRECARGADOS (2 GB RAM)
# ==========================================

MODELOS = {}
EXPLAINERS = {}   # SHAP TreeExplainer cargados permanentemente en memoria
SHAP_OK = False   
DATOS = {}
DATOS_ORIG = []
DF_COMPLETO = None  # Filas del CSV para analytics
DASHBOARD_RESERVAS = []  # todas las reservas del dashboard con predicción de riesgo
RESERVAS_SIMULADAS = []  # historial de simulaciones guardadas por usuario
FECHA_CORTE = '2023-01-01'

_RUTA_SIM = None  # se asigna tras conocer BASE_DIR (en cargar_datos_originales)


def calcular_riesgo(prob, segmento):
    u = UMBRALES_RIESGO.get(segmento, {'alto': 50, 'medio': 25})
    if prob > u['alto']:  return 'ALTO'
    if prob > u['medio']: return 'MEDIO'
    return 'BAJO'


def obtener_pais_agrupado_y_distancia(pais_seleccionado, hotel_id):
    """
    Calcula de forma dinámica la distancia real para hoteles en América:
    - Corto: Viajes nacionales (origen == país del hotel).
    - Medio: Viajes continentales dentro de América (USA, Canadá, Sudamérica...).
    - Largo: Viajes transatlánticos (Europa, Asia, etc.).
    """
    if not pais_seleccionado:
        return 'Otros', 'Largo'
        
    import unicodedata
    def normalizar(texto):
        return "".join(
            c for c in unicodedata.normalize('NFD', str(texto).strip().upper())
            if unicodedata.category(c) != 'Mn'
        ).replace('MEXICO', 'MEXICO').replace('ESPANA', 'ESPANA')
        
    pais_clean = normalizar(pais_seleccionado)
    
    # Mapeo del país donde está físicamente cada hotel ID
    HOTEL_PAIS = {
        6: 'MEXICO',      # Vallarta
        9: 'DOMINICANA',  # Dominican Fiesta
        83: 'BRASIL',     # Imbassai
        92: 'JAMAICA',    # Jamaica
        96: 'MEXICO',      # Costa Mujeres
        99: 'DOMINICANA',  # Cap Cana
        106: 'MEXICO',     # Riviera Maya
        107: 'DOMINICANA'  # Punta Cana
    }
    pais_hotel = HOTEL_PAIS.get(int(hotel_id), 'MEXICO')
    
    # Países del continente americano para clasificar distancia Media
    PAISES_AMERICA = {
        'MEXICO', 'ESTADOS UNIDOS', 'USA', 'CANADA', 'ARGENTINA', 'BRASIL', 
        'CHILE', 'COLOMBIA', 'PERU', 'URUGUAY', 'VENEZUELA', 'ECUADOR', 
        'JAMAICA', 'REPUBLICA DOMINICANA', 'DOMINICANA', 'SIN PAIS'
    }
    
    # 1. Obtener la agrupación histórica esperada por el encoder para el Target Encoding
    pais_agrup = 'Otros'
    for k, v in MAPA_PAIS_AGRUPADO.items():
        if normalizar(k) == pais_clean:
            pais_agrup = v
            break

    # 2. CLASIFICACIÓN DE DISTANCIA (Lógica América-céntrica):
    
    # Caso A: Vuelo nacional (Corto)
    is_mexico = (pais_clean in ('MEXICO', 'MEX') and pais_hotel == 'MEXICO')
    is_brasil = (pais_clean in ('BRASIL', 'BRA') and pais_hotel == 'BRASIL')
    is_jamaica = (pais_clean in ('JAMAICA', 'JAM') and pais_hotel == 'JAMAICA')
    is_dom = (pais_clean in ('DOMINICANA', 'REPUBLICA DOMINICANA', 'DOM') and pais_hotel == 'DOMINICANA')
    
    if is_mexico or is_brasil or is_jamaica or is_dom or (pais_clean == pais_hotel):
        distancia = 'Corto'
        
    # Caso B: Continental Americano (Medio)
    elif pais_clean in PAISES_AMERICA:
        distancia = 'Medio'
        
    # Caso C: Transatlántico (Largo)
    else:
        distancia = 'Largo'
        
    return pais_agrup, distancia


def cargar_modelos():
    """Carga permanentemente todos los pipelines y sus explainers SHAP en memoria (Máximo rendimiento)."""
    global SHAP_OK
    import sys
    import preprocesamiento as _prep
    import shap
    
    _main = sys.modules.get('__main__')
    if _main:
        for _name in dir(_prep):
            if not _name.startswith('_') and not hasattr(_main, _name):
                setattr(_main, _name, getattr(_prep, _name))
                
    try:
        # 1. Cargar los tres modelos XGBoost/RandomForest simultáneamente
        for segmento, config in SEGMENTOS.items():
            ruta_modelo = os.path.join(RUTA_MODELOS, config['modelo_archivo'])
            print(f"Pre-cargando modelo en RAM: {segmento}...")
            modelo = joblib.load(ruta_modelo)
            _limpiar_feature_names(modelo)
            MODELOS[segmento] = modelo
            
            # 2. Inicializar los TreeExplainers de SHAP permanentemente
            try:
                clf = modelo.named_steps['modelo']
                EXPLAINERS[segmento] = shap.TreeExplainer(clf)
                SHAP_OK = True
                print(f"OK: SHAP Explainer activo en memoria para {segmento}")
            except Exception as e_shap:
                print(f"AVISO: No se pudo crear explainer SHAP para {segmento}: {e_shap}")
                
        print("OK: Todos los modelos y estructuradores SHAP cargados con éxito en la RAM.")
    except Exception as e:
        print(f"ERROR crítico cargando modelos al iniciar: {e}")


# ==========================================
# TARGET ENCODING REAL
# ==========================================
MAPEOS_ENCODING = {}
try:
    import json as _json_map
    _RUTA_MAPEOS = os.path.join(BASE_DIR, 'mapeos_encoding.json')
    if os.path.exists(_RUTA_MAPEOS):
        with open(_RUTA_MAPEOS, encoding='utf-8') as _fm:
            MAPEOS_ENCODING = _json_map.load(_fm)
        print(f"OK: mapeos de encoding cargados ({[k for k in MAPEOS_ENCODING if not k.startswith('_')]})")
    else:
        print("AVISO: mapeos_encoding.json no encontrado — se usará la media global")
except Exception as e:
    print(f"AVISO: no se pudo cargar mapeos_encoding.json ({e})")

_MEDIA_GLOBAL_ENC = MAPEOS_ENCODING.get('_media_global', 0.38)

_PAIS_APP_A_ENCODER = {
    'España': 'ESPAÑA', 'Alemania': 'ALEMANIA', 'Argentina': 'ARGENTINA', 'Brasil': 'BRASIL',
    'Canada': 'CANADA', 'Mexico': 'MEXICO', 'USA': 'ESTADOS UNIDOS', 'Reino_Unido': 'REINO UNIDO',
    'Jamaica': 'JAMAICA',
}

def _cod_pais(pais_agrupado):
    m = MAPEOS_ENCODING.get('PAIS_AGRUPADO', {})
    if not m:
        return _MEDIA_GLOBAL_ENC
    clave = _PAIS_APP_A_ENCODER.get(pais_agrupado, str(pais_agrupado).strip().upper())
    return m.get(clave, m.get('OTRO', _MEDIA_GLOBAL_ENC))

def _cod_segmento(seg):
    m = MAPEOS_ENCODING.get('SEGMENTO', {})
    if not m:
        return _MEDIA_GLOBAL_ENC
    return m.get(str(seg).strip().upper(), _MEDIA_GLOBAL_ENC)

def _cod_habitacion(hab):
    m = MAPEOS_ENCODING.get('HABITACION_LIMPIA', {})
    if not m:
        return _MEDIA_GLOBAL_ENC
    return m.get(str(hab).strip().upper(), m.get('OTROS', _MEDIA_GLOBAL_ENC))

def _cargar_reservas_simuladas():
    global RESERVAS_SIMULADAS, _RUTA_SIM
    import json
    _RUTA_SIM = os.path.join(BASE_DIR, 'datos', 'reservas_simuladas.json')
    if os.path.exists(_RUTA_SIM):
        try:
            with open(_RUTA_SIM, 'r', encoding='utf-8') as f:
                RESERVAS_SIMULADAS = json.load(f)
        except Exception:
            RESERVAS_SIMULADAS = []

def _guardar_reservas_simuladas():
    import json
    if _RUTA_SIM:
        with open(_RUTA_SIM, 'w', encoding='utf-8') as f:
            json.dump(RESERVAS_SIMULADAS, f, ensure_ascii=False, indent=2)

def _limpiar_feature_names(estimador):
    for attr in ('feature_names_in_',):
        if hasattr(estimador, attr):
            try:
                delattr(estimador, attr)
            except Exception:
                pass
    if hasattr(estimador, 'steps'):
        for _, step in estimador.steps:
            _limpiar_feature_names(step)
    if hasattr(estimador, 'transformers_'):
        for _, t, _ in estimador.transformers_:
            _limpiar_feature_names(t)
    if hasattr(estimador, 'transformers'):
        for _, t, _ in estimador.transformers:
            _limpiar_feature_names(t)


# Nombres legibles y formateadores para las 20 columnas del pipeline
_NOMBRE_FEATURE = {
    'ANTELACION_DIAS': 'Antelación de la reserva',
    'ADR': 'Tarifa media (ADR)',
    'NOCHES': 'Número de noches',
    'NENES': 'Niños en la reserva',
    'BEBES': 'Bebés en la reserva',
    'SEGMENTO_COD': 'Segmento del hotel',
    'PAIS_AGRUPADO_COD': 'País de origen (codificado)',
    'HABITACION_LIMPIA_COD': 'Tipo de habitación (codificado)',
    'GRUPO_TIPO_COD': 'Reserva de grupo',
    'FUENTE_NEGOCIO_DIRECT SALES': 'Canal: Venta directa',
    'FUENTE_NEGOCIO_E-COMMERCE': 'Canal: E-Commerce / OTA',
    'FUENTE_NEGOCIO_OTHERS': 'Canal: Otros',
    'FUENTE_NEGOCIO_T.O. / T.A.': 'Canal: Tour Operador / Agencia',
    'PAX_TIPO_PAREJAS': 'Tipo de viajero: Pareja',
    'PAX_TIPO_SINGLE': 'Tipo de viajero: Individual',
    'TEMPORADA_BAJA': 'Temporada baja',
    'TEMPORADA_MEDIA': 'Temporada media',
    'DISTANCIA_LARGO': 'Origen de larga distancia',
    'DISTANCIA_MEDIO': 'Origen de media distancia',
    'DISTANCIA_NO INFO': 'Origen sin información',
}

def _valor_legible(col, features):
    v = features.get(col, 0)
    if col == 'ANTELACION_DIAS': return f'{int(v)} días'
    if col == 'ADR':             return f'${v:.0f}/noche'
    if col == 'NOCHES':          return f'{int(v)} noches'
    if col in ('NENES', 'BEBES'):return f'{int(v)}'
    if col in ('SEGMENTO_COD', 'PAIS_AGRUPADO_COD', 'HABITACION_LIMPIA_COD'):
        return f'{v:.3f}'
    return 'Sí' if float(v) >= 0.5 else 'No'


def _generar_explicacion_shap(features, segmento):
    """Explicación matemática basada estrictamente en los valores SHAP."""
    import numpy as np
    explainer = EXPLAINERS[segmento]
    X20 = np.array([[float(features.get(c, 0.0)) for c in COLUMNAS_PIPELINE]], dtype=np.float64)

    sv = explainer.shap_values(X20)
    arr = np.array(sv)
    if arr.ndim == 3:
        vals = arr[0, :, 1] if arr.shape[2] > 1 else arr[0, :, 0]
    elif arr.ndim == 2 and arr.shape[0] == 1:
        vals = arr[0]
    elif isinstance(sv, list):
        vals = np.array(sv[1])[0] if len(sv) > 1 else np.array(sv[0])[0]
    else:
        vals = arr.ravel()[:len(COLUMNAS_PIPELINE)]

    vals = np.asarray(vals, dtype=np.float64).ravel()[:len(COLUMNAS_PIPELINE)]
    max_abs = float(np.max(np.abs(vals))) or 1.0

    orden = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)
    factores = []
    for i in orden:
        col = COLUMNAS_PIPELINE[i]
        v = float(vals[i])
        peso = abs(v) / max_abs
        if peso < 0.05:
            continue
        if col in _NOMBRE_FEATURE and col not in ('ANTELACION_DIAS','ADR','NOCHES','NENES','BEBES',
                'SEGMENTO_COD','PAIS_AGRUPADO_COD','HABITACION_LIMPIA_COD'):
            if float(features.get(col, 0)) < 0.5 and peso < 0.15:
                continue
        direccion = 'sube' if v > 0 else ('baja' if v < 0 else 'neutro')
        impacto = 'alto' if peso >= 0.5 else ('medio' if peso >= 0.2 else 'bajo')
        factores.append({
            'factor': _NOMBRE_FEATURE.get(col, col),
            'valor': _valor_legible(col, features),
            'impacto': impacto,
            'direccion': direccion,
            'shap': round(v, 4),
            'peso_pct': round(peso * 100),
            'descripcion': f'Valor SHAP = {v:+.4f}',  # Diseño minimalista matemático solicitado
            'metodo': 'shap',
        })
        if len(factores) >= 6:
            break
    if not factores:
        raise ValueError('SHAP no produjo factores relevantes')
    return factores


def generar_explicacion(features, meta, segmento):
    """Punto de entrada único obligado a SHAP para garantizar consistencia científica."""
    return _generar_explicacion_shap(features, segmento)


def cargar_datos():
    """Carga los CSVs completos de entrenamiento directamente en la RAM (Gracias a los 2 GB)."""
    archivos = {
        'PEQUEÑO': 'df_pequeno.csv',
        'MEDIANO': 'df_mediano.csv',
        'GRANDE': 'df_grande.csv'
    }
    for segmento, archivo in archivos.items():
        ruta = os.path.join(RUTA_ENTRENAMIENTO, archivo)
        if os.path.exists(ruta):
            try:
                DATOS[segmento] = pd.read_csv(ruta, low_memory=False)
                print(f"OK: Dataset {segmento} cargado en memoria ({len(DATOS[segmento])} filas)")
            except Exception as e:
                print(f"ERROR leyendo {archivo}: {e}")
        else:
            print(f"AVISO: CSV no encontrado: {archivo}")


def _parse_date(s):
    if not s:
        return None
    s = str(s).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d',
                '%d-%m-%Y %H:%M:%S.%f', '%d-%m-%Y %H:%M:%S', '%d-%m-%Y',
                '%d/%m/%Y %H:%M:%S', '%d/%m/%Y'):
        try:
            return _dt.strptime(s[:len(fmt)], fmt)
        except Exception:
            continue
    try:
        return _dt.strptime(s[:10], '%Y-%m-%d')
    except Exception:
        try:
            return _dt.strptime(s[:10], '%d-%m-%Y')
        except Exception:
            return None


def _transformar_fila(row):
    moneda = str(row.get('MONEDA', '') or '').strip().upper()
    tc = TIPO_CAMBIO_USD.get(moneda, 1.0)
    valor_usd = (
        float(row.get('VALHAB',   0) or 0) +
        float(row.get('VALPEN',   0) or 0) +
        float(row.get('VALSERV',  0) or 0) +
        float(row.get('VALFIJOS', 0) or 0)
    ) / tc
    noches = max(int(row.get('NOCHES', 1) or 1), 1)
    adr = round(valor_usd / noches, 2)

    llegada_dt = _parse_date(row.get('LLEGADA', ''))
    toma_dt    = _parse_date(row.get('FECHA_TOMA', ''))
    try:
        antelacion = max(0, (llegada_dt - toma_dt).days) if (llegada_dt and toma_dt) else 0
    except Exception:
        antelacion = 0

    try:
        mes = llegada_dt.month if llegada_dt else 0
        temporada = TEMPORADA_MESES.get(mes, 'ALTA')
    except Exception:
        temporada = 'ALTA'

    pais_raw    = str(row.get('PAIS', '') or '').strip()
    hotel_id    = int(row.get('ID_HOTEL', 0) or 0)
    pais_agrup, distancia = obtener_pais_agrupado_y_distancia(pais_raw, hotel_id)

    pax = int(row.get('PAX', 2) or 2)
    pax_tipo = 'SINGLE' if pax == 1 else ('PAREJAS' if pax == 2 else 'FAMILIAS')

    try:
        grupo_cod = 1 if float(row.get('ID_MULTIPLE', 0) or 0) > 0 else 0
    except Exception:
        grupo_cod = 0

    segmento  = HOTEL_SIZE_MAPPING.get(hotel_id, 'GRANDE')

    seg_mercado = str(row.get('SEGMENTO', '') or '').strip()
    cod_segmento = _cod_segmento(seg_mercado)
    cod_pais     = _cod_pais(pais_agrup)
    cod_habitacion = _MEDIA_GLOBAL_ENC

    fuente = str(row.get('FUENTE_NEGOCIO', '') or '').strip()

    features = {
        'ANTELACION_DIAS': antelacion,
        'ADR':             adr,
        'NOCHES':          noches,
        'NENES':           int(row.get('NENES', 0) or 0),
        'BEBES':           int(row.get('BEBES', 0) or 0),
        'SEGMENTO_COD':           cod_segmento,
        'PAIS_AGRUPADO_COD':      cod_pais,
        'HABITACION_LIMPIA_COD':  cod_habitacion,
        'GRUPO_TIPO_COD':         grupo_cod,
        'FUENTE_NEGOCIO_DIRECT SALES':  1 if fuente == 'DIRECT SALES'  else 0,
        'FUENTE_NEGOCIO_E-COMMERCE':    1 if fuente == 'E-COMMERCE'    else 0,
        'FUENTE_NEGOCIO_OTHERS':        1 if fuente == 'OTHERS'        else 0,
        'FUENTE_NEGOCIO_T.O. / T.A.':  1 if fuente == 'T.O. / T.A.'  else 0,
        'PAX_TIPO_PAREJAS':  1 if pax_tipo == 'PAREJAS' else 0,
        'PAX_TIPO_SINGLE':   1 if pax_tipo == 'SINGLE'  else 0,
        'TEMPORADA_BAJA':    1 if temporada == 'BAJA'   else 0,
        'TEMPORADA_MEDIA':   1 if temporada == 'MEDIA'  else 0,
        'DISTANCIA_LARGO':   1 if distancia == 'Largo'    else 0,
        'DISTANCIA_MEDIO':   1 if distancia == 'Medio'    else 0,
        'DISTANCIA_NO INFO': 1 if distancia == 'No Info'  else 0,
    }
    meta = {
        'segmento': segmento, 'adr': adr, 'temporada': temporada,
        'distancia': distancia, 'pais_agrup': pais_agrup, 'pax_tipo': pax_tipo,
        'antelacion': antelacion,
    }
    return features, meta


def cargar_datos_originales():
    global DATOS_ORIG, DF_COMPLETO
    ruta = os.path.join(RUTA_DATOS, 'Reservas_22_23.csv')
    if not os.path.exists(ruta):
        print("AVISO: Reservas_22_23.csv no encontrado")
        return
    try:
        cols = ['ID_RESERVA', 'ID_HOTEL', 'LLEGADA', 'FECHA_TOMA', 'NOCHES',
                'PAX', 'ADULTOS', 'NENES', 'BEBES', 'PAIS', 'FUENTE_NEGOCIO',
                'TIPO', 'STATUS', 'ID_MULTIPLE', 'MONEDA', 'SEGMENTO',
                'VALHAB', 'VALPEN', 'VALSERV', 'VALFIJOS']
        # Con 2 GB cargamos la muestra estándar de 10.000 filas para el dashboard
        df = pd.read_csv(ruta, sep=';', nrows=10000, usecols=cols,
                         on_bad_lines='skip', low_memory=False)
        print(f"OK: CSV original leído ({len(df)} filas)")
        df['ID_HOTEL'] = pd.to_numeric(df['ID_HOTEL'], errors='coerce').fillna(0).astype(int)
        df = df[df['ID_HOTEL'].isin(HOTEL_SIZE_MAPPING.keys())]
        DF_COMPLETO = df
        muestra = df.sample(min(300, len(df)), random_state=42).reset_index(drop=True)
        print(f"OK: Muestra seleccionada ({len(muestra)} filas)")

        import numpy as np

        def _predecir(modelo, features_dict):
            clf = modelo.named_steps['modelo']
            X20 = np.array([[float(features_dict.get(c, 0.0)) for c in COLUMNAS_PIPELINE]],
                           dtype=np.float64)
            return clf.predict_proba(X20)

        registros = []
        for _, row in muestra.iterrows():
            try:
                features, meta = _transformar_fila(row)
            except Exception:
                continue

            prob = None
            riesgo = None
            seg_key = meta['segmento']
            if seg_key in MODELOS:
                try:
                    prob_arr = _predecir(MODELOS[seg_key], features)
                    prob = round(float(prob_arr[0][1]) * 100, 1)
                    riesgo = calcular_riesgo(prob, seg_key)
                except Exception:
                    pass

            status_val = int(row.get('STATUS', 0) or 0)
            cancelada  = bool(MAPA_STATUS.get(status_val, 0))
            hotel_id   = int(row.get('ID_HOTEL', 0))
            hotel_nom  = HOTEL_NOMBRES.get(hotel_id, f'Hotel {hotel_id}')
            
            # SHAP real e innegociable para todas las filas del histórico
            explicacion = _generar_explicacion_shap(features, seg_key)

            registros.append({
                'id_reserva':  str(row.get('ID_RESERVA', '')),
                'hotel_id':    hotel_id,
                'hotel_nom':   hotel_nom,
                'segmento':    seg_key,
                'llegada':     str(row.get('LLEGADA', '') or '')[:10],
                'tipo':        str(row.get('TIPO', '') or ''),
                'adr':         meta['adr'],
                'noches':      int(row.get('NOCHES', 1) or 1),
                'antelacion':  meta['antelacion'],
                'adultos':     int(row.get('ADULTOS', 0) or 0),
                'nenes':       int(row.get('NENES', 0) or 0),
                'bebes':       int(row.get('BEBES', 0) or 0),
                'pais_orig':   str(row.get('PAIS', '') or '').title(),
                'pais_agrup':  meta['pais_agrup'],
                'fuente':      str(row.get('FUENTE_NEGOCIO', '') or ''),
                'pax_tipo':    meta['pax_tipo'],
                'temporada':   meta['temporada'],
                'distancia':   meta['distancia'],
                'cancelada':   cancelada,
                'estado':      'CANCELADA' if cancelada else 'ACTIVA',
                'prob_pred':   prob,
                'riesgo_pred': riesgo,
                'explicacion': explicacion,
                'metodo_exp':  'shap',
                '_features':   features,
            })

        DATOS_ORIG = registros
        preds_ok = sum(1 for r in registros if r['prob_pred'] is not None)
        print(f"OK: {len(DATOS_ORIG)} reservas cargadas con explicaciones SHAP reales.")
        
        global FECHA_CORTE
        fechas = sorted([r['llegada'] for r in registros if r.get('llegada') and len(r.get('llegada', '')) >= 7])
        FECHA_CORTE = fechas[len(fechas) // 2] if fechas else '2023-01-01'
        print(f"OK: Fecha de corte calculada: {FECHA_CORTE}")
    except Exception as e:
        import traceback
        print(f"ERROR al cargar datos originales: {e}")
        traceback.print_exc()


def cargar_dashboard():
    global DASHBOARD_RESERVAS
    if DF_COMPLETO is None or len(DF_COMPLETO) == 0:
        print("AVISO: sin DF_COMPLETO para el dashboard")
        return
    import numpy as np
    registros = []
    feats_por_seg = {seg: [] for seg in MODELOS.keys()}
    idx_por_seg = {seg: [] for seg in MODELOS.keys()}
    for _, row in DF_COMPLETO.iterrows():
        try:
            features, meta = _transformar_fila(row)
        except Exception:
            continue
        llegada = str(row.get('LLEGADA', '') or '')[:10]
        if len(llegada) < 7:
            continue
        hotel_id  = int(float(row.get('ID_HOTEL', 0) or 0))
        hotel_nom = HOTEL_NOMBRES.get(hotel_id)
        if not hotel_nom:
            continue
        status    = int(float(row.get('STATUS', 0) or 0))
        cancelada = bool(MAPA_STATUS.get(status, 0))
        moneda    = str(row.get('MONEDA', '') or '').strip().upper()
        tc        = TIPO_CAMBIO_USD.get(moneda, 1.0)
        ingresos  = (float(row.get('VALHAB', 0) or 0) + float(row.get('VALPEN', 0) or 0) +
                     float(row.get('VALSERV', 0) or 0) + float(row.get('VALFIJOS', 0) or 0)) / tc
        seg = meta['segmento']
        reg = {'mes': llegada[:7], 'llegada': llegada, 'hotel': hotel_nom,
               'cancelada': cancelada, 'ingresos': round(ingresos), 'prob': None, 'riesgo': None}
        if seg in feats_por_seg:
            feats_por_seg[seg].append([float(features.get(c, 0.0)) for c in COLUMNAS_PIPELINE])
            idx_por_seg[seg].append(len(registros))
        registros.append(reg)

    for seg, modelo in MODELOS.items():
        if not feats_por_seg[seg]:
            continue
        try:
            clf   = modelo.named_steps['modelo']
            X     = np.array(feats_por_seg[seg], dtype=np.float64)
            probs = clf.predict_proba(X)[:, 1]
            for j, idx in enumerate(idx_por_seg[seg]):
                p = round(float(probs[j]) * 100, 1)
                registros[idx]['prob']   = p
                registros[idx]['riesgo'] = calcular_riesgo(p, seg)
        except Exception as e:
            print(f"AVISO: no se pudo predecir el dashboard para {seg}: {e}")

    DASHBOARD_RESERVAS = registros
    print(f"OK: Dashboard — {len(DASHBOARD_RESERVAS)} reservas procesadas con éxito.")


def agregar_dashboard(corte):
    from collections import defaultdict
    def _blanco():
        return {'pas_activas': 0, 'pas_canceladas': 0, 'ing_ganados': 0.0, 'ing_perdidos': 0.0,
                'fut_alto': 0, 'fut_medio': 0, 'fut_bajo': 0,
                'ing_alto': 0.0, 'ing_medio': 0.0, 'ing_bajo': 0.0}
    meses = defaultdict(lambda: {'stats': _blanco(), 'hoteles': defaultdict(_blanco)})

    for r in DASHBOARD_RESERVAS:
        nodo = meses[r['mes']]
        s = nodo['stats']; h = nodo['hoteles'][r['hotel']]
        ing = r['ingresos']
        if r['llegada'] <= corte:
            if r['cancelada']:
                s['pas_canceladas'] += 1; s['ing_perdidos'] += ing
                h['pas_canceladas'] += 1; h['ing_perdidos'] += ing
            else:
                s['pas_activas'] += 1; s['ing_ganados'] += ing
                h['pas_activas'] += 1; h['ing_ganados'] += ing
        else:
            rg = r['riesgo'] or 'BAJO'
            k = 'alto' if rg == 'ALTO' else ('medio' if rg == 'MEDIO' else 'bajo')
            s['fut_' + k] += 1; s['ing_' + k] += ing
            h['fut_' + k] += 1; h['ing_' + k] += ing

    salida = []
    for mes in sorted(meses.keys()):
        nodo = meses[mes]; s = nodo['stats']
        hoteles = []
        for hn, h in nodo['hoteles'].items():
            item = {'hotel': hn}
            item.update({k: (round(v) if 'ing_' in k else v) for k, v in h.items()})
            item['total'] = (h['pas_activas'] + h['pas_canceladas'] +
                             h['fut_alto'] + h['fut_medio'] + h['fut_bajo'])
            hoteles.append(item)
        hoteles.sort(key=lambda x: -x['total'])
        fila = {'mes': mes}
        fila.update({k: (round(v) if 'ing_' in k else v) for k, v in s.items()})
        fila['total'] = (s['pas_activas'] + s['pas_canceladas'] +
                         s['fut_alto'] + s['fut_medio'] + s['fut_bajo'])
        fila['hoteles'] = hoteles
        salida.append(fila)
    return salida


# PRECARGA PERMANENTE EN INICIO (MÁXIMO RENDIMIENTO)
cargar_modelos()
cargar_datos()
cargar_datos_originales()
cargar_dashboard()
_cargar_reservas_simuladas()


def _agregar_estadisticas():
    from collections import defaultdict
    if DF_COMPLETO is None or len(DF_COMPLETO) == 0:
        return {'por_mes': [], 'por_hotel': []}

    por_mes       = defaultdict(lambda: {'total': 0, 'canceladas': 0, 'ingresos': 0.0})
    por_hotel     = defaultdict(lambda: {'total': 0, 'canceladas': 0, 'ingresos': 0.0})
    por_mes_hotel = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'canceladas': 0, 'ingresos': 0.0}))

    for _, row in DF_COMPLETO.iterrows():
        mes_str = str(row.get('LLEGADA', '') or '')[:7]
        if not mes_str or mes_str == 'nan' or len(mes_str) < 7:
            continue

        hotel_id  = int(float(row.get('ID_HOTEL', 0) or 0))
        hotel_nom = HOTEL_NOMBRES.get(hotel_id)
        if not hotel_nom:
            continue

        status    = int(float(row.get('STATUS', 0) or 0))
        cancelada = bool(MAPA_STATUS.get(status, 0))

        moneda   = str(row.get('MONEDA', '') or '').strip().upper()
        tc       = TIPO_CAMBIO_USD.get(moneda, 1.0)
        ingresos = (
            float(row.get('VALHAB',   0) or 0) +
            float(row.get('VALPEN',   0) or 0) +
            float(row.get('VALSERV',  0) or 0) +
            float(row.get('VALFIJOS', 0) or 0)
        ) / tc

        por_mes[mes_str]['total']    += 1
        por_mes[mes_str]['ingresos'] += ingresos
        if cancelada:
            por_mes[mes_str]['canceladas'] += 1

        por_hotel[hotel_nom]['total']    += 1
        por_hotel[hotel_nom]['ingresos'] += ingresos
        if cancelada:
            por_hotel[hotel_nom]['canceladas'] += 1

        por_mes_hotel[mes_str][hotel_nom]['total']    += 1
        por_mes_hotel[mes_str][hotel_nom]['ingresos'] += ingresos
        if cancelada:
            por_mes_hotel[mes_str][hotel_nom]['canceladas'] += 1

    meses = sorted(por_mes.keys())
    return {
        'por_mes': [
            {'mes': m, 'total': por_mes[m]['total'],
             'canceladas': por_mes[m]['canceladas'],
             'activas': por_mes[m]['total'] - por_mes[m]['canceladas'],
             'ingresos': round(por_mes[m]['ingresos']),
             'hoteles': sorted([
                 {'hotel': h,
                  'total': v['total'],
                  'canceladas': v['canceladas'],
                  'activas': v['total'] - v['canceladas'],
                  'ingresos': round(v['ingresos'])}
                 for h, v in por_mes_hotel[m].items()
             ], key=lambda x: -x['total'])}
            for m in meses
        ],
        'por_hotel': sorted([
            {'hotel': h, 'total': v['total'],
             'canceladas': v['canceladas'],
             'activas': v['total'] - v['canceladas'],
             'ingresos': round(v['ingresos'])}
            for h, v in por_hotel.items()
        ], key=lambda x: x['total'], reverse=True),
    }


# ==========================================
# RUTAS DE AUTENTICACIÓN
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        usuario = request.form.get('usuario')
        password = request.form.get('password')

        if usuario in USUARIOS and USUARIOS[usuario] == password:
            session['usuario'] = usuario
            session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return redirect(url_for('estadisticas'))
        else:
            return render_template('login.html', error='Usuario o contraseña incorrectos')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ==========================================
# PROTECCIÓN DE RUTAS
# ==========================================

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'usuario' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==========================================
# PÁGINA DE INICIO (LANDING)
# ==========================================

HOTEL_INFO = {
    6:   {'nom': 'Palladium Vallarta',                      'loc': 'Puerto Vallarta, México',  'pais': '🇲🇽', 'estrellas': 5, 'precio': 216, 'tag': 'Todo incluido', 'grad': 'linear-gradient(160deg,#0d7377,#14a085)', 'desc': 'Resort frente al Pacífico mexicano.'},
    9:   {'nom': 'Dominican Fiesta Hotel & Casino',         'loc': 'Santo Domingo, Rep. Dom.', 'pais': '🇩🇴', 'estrellas': 4, 'precio': 100, 'tag': 'Hotel & Casino', 'grad': 'linear-gradient(160deg,#c0392b,#8e44ad)', 'desc': 'Hotel y casino en la capital dominicana.'},
    83:  {'nom': 'Grand Palladium Imbassai Resort & Spa',   'loc': 'Bahía, Brasil',            'pais': '🇧🇷', 'estrellas': 5, 'precio': 392, 'tag': 'Todo incluido', 'grad': 'linear-gradient(160deg,#00b09b,#1e3c72)', 'desc': 'Resort entre la selva atlántica del nordeste brasileño.'},
    92:  {'nom': 'Grand Palladium Jamaica & Lady Hamilton', 'loc': 'Montego Bay, Jamaica',     'pais': '🇯🇲', 'estrellas': 5, 'precio': 235, 'tag': 'Todo incluido', 'grad': 'linear-gradient(160deg,#1a6b3c,#f9ca24)', 'desc': 'Resort caribeño en la bahía de Montego Bay.'},
    96:  {'nom': 'Complejo Costa Mujeres',                  'loc': 'Costa Mujeres, México',    'pais': '🇲🇽', 'estrellas': 5, 'precio': 305, 'tag': 'Luxury', 'grad': 'linear-gradient(160deg,#0099f7,#00d2d3)', 'desc': 'Complejo frente a aguas turquesas a minutos de Cancún.'},
    99:  {'nom': 'TRS Cap Cana',                            'loc': 'Cap Cana, Rep. Dom.',      'pais': '🇩🇴', 'estrellas': 5, 'precio': 267, 'tag': 'Adults only', 'grad': 'linear-gradient(160deg,#0f0c29,#302b63)', 'desc': 'Resort exclusivo solo para adultos en Cap Cana.'},
    106: {'nom': 'Complejo Riviera Maya',                   'loc': 'Playa del Carmen, México', 'pais': '🇲🇽', 'estrellas': 5, 'precio': 242, 'tag': 'Todo incluido', 'grad': 'linear-gradient(160deg,#11998e,#38ef7d)', 'desc': 'Complejo en la Riviera Maya: selva, cenotes y playa.'},
    107: {'nom': 'Complejo Punta Cana',                     'loc': 'Punta Cana, Rep. Dom.',    'pais': '🇩🇴', 'estrellas': 5, 'precio': 180, 'tag': 'Todo incluido', 'grad': 'linear-gradient(160deg,#005c97,#363795)', 'desc': 'Complejo de playa en Punta Cana.'},
}

@app.route('/')
def index():
    return render_template('booking.html', hoteles=HOTEL_INFO)


@app.route('/api/predecir_publico', methods=['POST'])
def predecir_publico():
    import numpy as np
    data = request.get_json() or {}

    hotel_id  = int(data.get('hotel_id', 107))
    segmento  = HOTEL_SIZE_MAPPING.get(hotel_id, 'GRANDE')
    hotel_nom = HOTEL_NOMBRES.get(hotel_id, 'Hotel')

    llegada_dt = _parse_date(data.get('llegada', ''))
    toma_dt    = _parse_date(data.get('fecha_toma', ''))
    salida_dt  = _parse_date(data.get('salida', ''))
    try:
        antelacion = max(0, (llegada_dt - toma_dt).days) if (llegada_dt and toma_dt) else 0
    except Exception:
        antelacion = 0
    try:
        noches = max(1, (salida_dt - llegada_dt).days) if (salida_dt and llegada_dt) else 1
    except Exception:
        noches = 1

    adr      = float(data.get('adr', HOTEL_INFO.get(hotel_id, {}).get('precio', 200)) or 200)
    nenes    = int(data.get('nenes', 0) or 0)
    bebes    = int(data.get('bebes', 0) or 0)
    adultos  = int(data.get('adultos', 2) or 2)
    pais_raw = str(data.get('pais', '') or '').strip()

    mes       = llegada_dt.month if llegada_dt else 6
    temporada = TEMPORADA_MESES.get(mes, 'ALTA')
    pais_agrup, distancia = obtener_pais_agrupado_y_distancia(pais_raw, hotel_id)
    
    pax        = adultos + nenes
    pax_tipo   = 'SINGLE' if pax == 1 else ('PAREJAS' if pax == 2 else 'FAMILIAS')
    cod_pais = _cod_pais(pais_agrup)

    features = {
        'ANTELACION_DIAS':              antelacion,
        'ADR':                          adr,
        'NOCHES':                       noches,
        'NENES':                        nenes,
        'BEBES':                        bebes,
        'SEGMENTO_COD':                 _MEDIA_GLOBAL_ENC,
        'PAIS_AGRUPADO_COD':            cod_pais,
        'HABITACION_LIMPIA_COD':        _MEDIA_GLOBAL_ENC,
        'GRUPO_TIPO_COD':               0,
        'FUENTE_NEGOCIO_DIRECT SALES':  0,
        'FUENTE_NEGOCIO_E-COMMERCE':    1,
        'FUENTE_NEGOCIO_OTHERS':        0,
        'FUENTE_NEGOCIO_T.O. / T.A.':  0,
        'PAX_TIPO_PAREJAS':  1 if pax_tipo == 'PAREJAS' else 0,
        'PAX_TIPO_SINGLE':   1 if pax_tipo == 'SINGLE'  else 0,
        'TEMPORADA_BAJA':    1 if temporada == 'BAJA'   else 0,
        'TEMPORADA_MEDIA':   1 if temporada == 'MEDIA'  else 0,
        'DISTANCIA_LARGO':   1 if distancia == 'Largo'  else 0,
        'DISTANCIA_MEDIO':   1 if distancia == 'Medio'  else 0,
        'DISTANCIA_NO INFO': 1 if distancia == 'No Info' else 0,
    }

    if segmento not in MODELOS:
        return jsonify({'error': 'Modelo no disponible'}), 400

    clf  = MODELOS[segmento].named_steps['modelo']
    X20  = np.array([[float(features.get(c, 0.0)) for c in COLUMNAS_PIPELINE]], dtype=np.float64)
    prob = round(float(clf.predict_proba(X20)[0][1]) * 100, 1)
    riesgo = calcular_riesgo(prob, segmento)

    total_precio = round(adr * noches * adultos, 2)

    from datetime import datetime as _dt3
    meta_h = {'segmento': segmento, 'adr': adr, 'temporada': temporada,
              'distancia': distancia, 'pais_agrup': pais_agrup,
              'pax_tipo': pax_tipo, 'antelacion': antelacion}
    sim_id = _dt3.now().strftime('%Y%m%d_%H%M%S_%f')
    pais_raw_orig = str(data.get('pais', '') or '')
    registro_pub = {
        'id':           sim_id,
        'usuario':      'publico',
        'tipo':         'cliente',
        'fecha_sim':    _dt3.now().strftime('%d/%m/%Y %H:%M'),
        'hotel_id':     hotel_id,
        'hotel_nom':    hotel_nom,
        'segmento':     segmento,
        'llegada':      data.get('llegada', ''),
        'salida':       data.get('salida', ''),
        'noches':       noches,
        'adultos':      adultos,
        'nenes':        nenes,
        'bebes':        0,
        'pais':         pais_raw_orig,
        'adr':          round(adr, 2),
        'es_grupo':     False,
        'antelacion':   antelacion,
        'temporada':    temporada,
        'distancia':    distancia,
        'pax_tipo':     pax_tipo,
        'prob':         prob,
        'riesgo':       riesgo,
        'explicacion':  _generar_explicacion_shap(features, segmento),  # SHAP Real
    }
    RESERVAS_SIMULADAS.append(registro_pub)
    _guardar_reservas_simuladas()

    return jsonify({
        'prob':         prob,
        'riesgo':       riesgo,
        'hotel_nom':    hotel_nom,
        'noches':       noches,
        'antelacion':   antelacion,
        'temporada':    temporada,
        'total_precio': total_precio,
        'adr':          adr,
    })




# ==========================================
# DASHBOARD / ESTADÍSTICAS
# ==========================================

@app.route('/estadisticas')
@login_required
def estadisticas():
    usuario = session.get('usuario')

    segmentos_info = {seg: {'algoritmo': cfg['algoritmo'], 'auc_roc': cfg['auc_roc']}
                      for seg, cfg in SEGMENTOS.items()}
    datos_orig_full = [{k: v for k, v in r.items() if k != '_features'} for r in DATOS_ORIG]
    hoteles_disponibles = sorted(set(r['hotel_nom'] for r in DATOS_ORIG if r.get('hotel_nom')))
    analytics = _agregar_estadisticas()

    sims_usuario = [r for r in RESERVAS_SIMULADAS
                    if r.get('usuario') == usuario or r.get('tipo') == 'cliente']
    sims_usuario = sorted(sims_usuario, key=lambda x: x.get('id',''), reverse=True)

    if not DATOS:
        return render_template('estadisticas.html', usuario=usuario,
                               totales=None, por_segmento={}, reservas=[],
                               datos_orig=datos_orig_full, segmentos_info=segmentos_info,
                               hoteles_disponibles=hoteles_disponibles,
                               analytics=analytics,
                               reservas_simuladas=sims_usuario,
                               fecha_corte=FECHA_CORTE,
                               umbrales_riesgo=UMBRALES_RIESGO,
                               shap_activo=SHAP_OK)

    totales = {
        'reservas':   sum(len(df) for df in DATOS.values()),
        'canceladas': sum(int(df['STATUS_BOOL'].sum()) for df in DATOS.values()),
    }
    totales['tasa'] = round(totales['canceladas'] / totales['reservas'] * 100, 2)

    por_segmento = {}
    for seg, df in DATOS.items():
        por_segmento[seg] = {
            'total':      len(df),
            'canceladas': int(df['STATUS_BOOL'].sum()),
            'tasa':       round(float(df['STATUS_BOOL'].mean()) * 100, 2),
            'algoritmo':  SEGMENTOS[seg]['algoritmo'],
            'auc_roc':    SEGMENTOS[seg]['auc_roc'],
        }

    def get_ohe(row, prefix, baseline):
        for col in row.index:
            if col.startswith(prefix) and float(row[col]) == 1.0:
                return col[len(prefix):]
        return baseline

    SAMPLE = 300
    reservas = []
    for seg, df in DATOS.items():
        muestra = df.sample(min(SAMPLE, len(df)), random_state=42).reset_index(drop=True)

        for i, row in muestra.iterrows():
            cancelada = bool(row.get('STATUS_BOOL', False))
            prob = None
            riesgo_pred = None
            reservas.append({
                'seg':        seg,
                'adr':        round(float(row.get('ADR', 0)), 0),
                'noches':     int(row.get('NOCHES', 0)),
                'antelacion': int(row.get('ANTELACION_DIAS', 0)),
                'fuente':     get_ohe(row, 'FUENTE_NEGOCIO_', 'Corporate'),
                'temporada':  get_ohe(row, 'TEMPORADA_', 'Alta'),
                'distancia':  get_ohe(row, 'DISTANCIA_', 'Corto'),
                'pax_tipo':   get_ohe(row, 'PAX_TIPO_', 'Familias'),
                'grupo':      'Grupo' if int(row.get('GRUPO_TIPO_COD', 0)) == 1 else 'Individual',
                'cancelada':  cancelada,
                'estado':     'CANCELADA' if cancelada else 'ACTIVA',
                'prob_pred':  prob,
                'riesgo_pred': riesgo_pred,
            })

    return render_template('estadisticas.html', usuario=usuario,
                           totales=totales, por_segmento=por_segmento,
                           reservas=reservas, datos_orig=datos_orig_full,
                           segmentos_info=segmentos_info,
                           hoteles_disponibles=hoteles_disponibles,
                           analytics=analytics,
                           reservas_simuladas=sims_usuario,
                           fecha_corte=FECHA_CORTE,
                           umbrales_riesgo=UMBRALES_RIESGO)


# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/nueva-reserva')
@login_required
def nueva_reserva():
    usuario = session.get('usuario')
    hoteles = [{'id': hid, 'nombre': nom, 'segmento': HOTEL_SIZE_MAPPING.get(hid, 'GRANDE')}
               for hid, nom in sorted(HOTEL_NOMBRES.items(), key=lambda x: x[1])]
               
    # Deduplica y unifica la lista de países para el desplegable (México con acento)
    paises_unicos = set()
    for p in MAPA_PAIS_AGRUPADO.keys():
        p_clean = p.strip().title()
        if p_clean == "Mexico": p_clean = "México"
        paises_unicos.add(p_clean)
    paises = sorted(list(paises_unicos))

    adr_por_hotel = {}
    for r in DATOS_ORIG:
        hid = r.get('hotel_id')
        adr = r.get('adr', 0)
        if hid and adr and adr > 0:
            if hid not in adr_por_hotel:
                adr_por_hotel[hid] = []
            adr_por_hotel[hid].append(adr)
    adr_medias = {hid: round(sum(v) / len(v), 2) for hid, v in adr_por_hotel.items()}

    segmentos_mkt = sorted(MAPEOS_ENCODING.get('SEGMENTO', {}).keys())

    return render_template('nueva_reserva.html', usuario=usuario,
                           hoteles=hoteles, paises=paises, adr_medias=adr_medias,
                           segmentos_mkt=segmentos_mkt)


@app.route('/api/predecir', methods=['POST'])
@login_required
def api_predecir():
    import numpy as np
    data = request.get_json() or {}

    hotel_id   = int(data.get('hotel_id', 0))
    segmento   = HOTEL_SIZE_MAPPING.get(hotel_id, 'GRANDE')
    hotel_nom  = HOTEL_NOMBRES.get(hotel_id, f'Hotel {hotel_id}')

    llegada_dt = _parse_date(data.get('llegada', ''))
    toma_dt    = _parse_date(data.get('fecha_toma', ''))
    try:
        antelacion = max(0, (llegada_dt - toma_dt).days) if (llegada_dt and toma_dt) else 0
    except Exception:
        antelacion = 0

    salida_dt = _parse_date(data.get('salida', ''))
    try:
        noches = max(1, (salida_dt - llegada_dt).days) if (salida_dt and llegada_dt) else max(int(data.get('noches', 1) or 1), 1)
    except Exception:
        noches = max(int(data.get('noches', 1) or 1), 1)

    adr       = float(data.get('adr', 0) or 0)
    nenes     = int(data.get('nenes', 0) or 0)
    bebes     = int(data.get('bebes', 0) or 0)
    adultos   = int(data.get('adultos', 2) or 2)
    grupo_cod = 1 if data.get('es_grupo', False) else 0
    fuente    = 'E-COMMERCE'
    pais_raw  = str(data.get('pais', '') or '').strip()

    pais_agrup, distancia = obtener_pais_agrupado_y_distancia(pais_raw, hotel_id)

    mes        = llegada_dt.month if llegada_dt else 0
    temporada  = TEMPORADA_MESES.get(mes, 'ALTA')
    pax        = adultos + nenes
    pax_tipo   = 'SINGLE' if pax == 1 else ('PAREJAS' if pax == 2 else 'FAMILIAS')
    segmento_sel = str(data.get('segmento_mkt', '') or '').strip()
    cod_pais     = _cod_pais(pais_agrup)
    cod_seg_mkt  = _cod_segmento(segmento_sel) if segmento_sel else _MEDIA_GLOBAL_ENC

    features = {
        'ANTELACION_DIAS':              antelacion,
        'ADR':                          adr,
        'NOCHES':                       noches,
        'NENES':                        nenes,
        'BEBES':                        bebes,
        'SEGMENTO_COD':                 cod_seg_mkt,
        'PAIS_AGRUPADO_COD':            cod_pais,
        'HABITACION_LIMPIA_COD':        _MEDIA_GLOBAL_ENC,
        'GRUPO_TIPO_COD':               grupo_cod,
        'FUENTE_NEGOCIO_DIRECT SALES':  1 if fuente == 'DIRECT SALES'  else 0,
        'FUENTE_NEGOCIO_E-COMMERCE':    1 if fuente == 'E-COMMERCE'    else 0,
        'FUENTE_NEGOCIO_OTHERS':        1 if fuente == 'OTHERS'        else 0,
        'FUENTE_NEGOCIO_T.O. / T.A.':  1 if fuente == 'T.O. / T.A.'  else 0,
        'PAX_TIPO_PAREJAS':  1 if pax_tipo == 'PAREJAS' else 0,
        'PAX_TIPO_SINGLE':   1 if pax_tipo == 'SINGLE'  else 0,
        'TEMPORADA_BAJA':    1 if temporada == 'BAJA'   else 0,
        'TEMPORADA_MEDIA':   1 if temporada == 'MEDIA'  else 0,
        'DISTANCIA_LARGO':   1 if distancia == 'Largo'  else 0,
        'DISTANCIA_MEDIO':   1 if distancia == 'Medio'  else 0,
        'DISTANCIA_NO INFO': 1 if distancia == 'No Info' else 0,
    }
    meta = {
        'segmento': segmento, 'adr': adr, 'temporada': temporada,
        'distancia': distancia, 'pais_agrup': pais_agrup,
        'pax_tipo': pax_tipo, 'antelacion': antelacion,
    }

    if segmento not in MODELOS:
        return jsonify({'error': 'Modelo no disponible'}), 400

    clf  = MODELOS[segmento].named_steps['modelo']
    X20  = np.array([[float(features.get(c, 0.0)) for c in COLUMNAS_PIPELINE]], dtype=np.float64)
    prob = round(float(clf.predict_proba(X20)[0][1]) * 100, 1)
    riesgo = calcular_riesgo(prob, segmento)
    
    # SHAP real e inmediato para el simulador de reservas
    explicacion = _generar_explicacion_shap(features, segmento)

    from datetime import datetime as _dt2
    sim_id = _dt2.now().strftime('%Y%m%d_%H%M%S_%f')
    registro_sim = {
        'id':               sim_id,
        'usuario':          session.get('usuario', ''),
        'fecha_sim':        _dt2.now().strftime('%d/%m/%Y %H:%M'),
        'hotel_id':         hotel_id,
        'hotel_nom':        hotel_nom,
        'segmento':         segmento,
        'llegada':          data.get('llegada', ''),
        'salida':           data.get('salida', ''),
        'noches':           noches,
        'adultos':          adultos,
        'nenes':            nenes,
        'bebes':            bebes,
        'pais':             data.get('pais', ''),
        'adr':              round(adr, 2),
        'es_grupo':         bool(data.get('es_grupo', False)),
        'antelacion':       antelacion,
        'temporada':        temporada,
        'distancia':        distancia,
        'pax_tipo':         pax_tipo,
        'prob':             prob,
        'riesgo':           riesgo,
        'explicacion':      explicacion,
    }
    RESERVAS_SIMULADAS.append(registro_sim)
    _guardar_reservas_simuladas()

    return jsonify({
        'prob':        prob,
        'riesgo':      riesgo,
        'sim_id':      sim_id,
        'segmento':    segmento,
        'hotel_nom':   hotel_nom,
        'antelacion':  antelacion,
        'temporada':   temporada,
        'distancia':   distancia,
        'pax_tipo':    pax_tipo,
        'explicacion': explicacion,
    })


@app.route('/api/reservas_simuladas/eliminar/<sim_id>', methods=['DELETE'])
@login_required
def eliminar_reserva_simulada(sim_id):
    global RESERVAS_SIMULADAS
    usuario = session.get('usuario', '')
    RESERVAS_SIMULADAS = [r for r in RESERVAS_SIMULADAS
                          if not (r['id'] == sim_id and r['usuario'] == usuario)]
    _guardar_reservas_simuladas()
    return jsonify({'ok': True})


@app.route('/api/dashboard')
@login_required
def api_dashboard():
    corte = request.args.get('corte') or FECHA_CORTE
    return jsonify({'corte': corte, 'meses': agregar_dashboard(corte)})


@app.route('/api/datos_segmento/<segmento>')
@login_required
def api_datos_segmento(segmento):
    if segmento not in DATOS:
        return jsonify({'error': 'Datos no disponibles para este segmento'}), 404

    df = DATOS[segmento]
    return jsonify({
        'total': len(df),
        'canceladas': int(df['STATUS_BOOL'].sum()),
        'tasa': round(df['STATUS_BOOL'].sum() / len(df) * 100, 2)
    })

# ==========================================
# EJECUTAR APLICACIÓN
# ==========================================

if __name__ == '__main__':
    puerto = int(os.environ.get('PORT', FLASK_PORT))
    print(f"\n{'='*60}")
    print("APLICACIÓN PALLADIUM - PREDICTOR DE CANCELACIONES (PREMIUM 2GB)")
    print(f"{'='*60}")
    print(f"Escuchando en el puerto: {puerto}")
    print(f"Modelos pre-cargados en RAM: {list(MODELOS.keys())}")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=puerto, debug=False)
