# ==========================================
# APLICACIÓN FLASK - PREDICTOR PALLADIUM
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
# CARGAR MODELOS AL INICIAR
# ==========================================

MODELOS = {}
DATOS = {}
DATOS_ORIG = []
DF_COMPLETO = None  # 10 000 filas del CSV para analytics
RESERVAS_SIMULADAS = []  # historial de simulaciones guardadas por usuario

_RUTA_SIM = None  # se asigna tras conocer BASE_DIR (en cargar_datos_originales)

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
    """Elimina feature_names_in_ de todos los pasos para evitar validación estricta de sklearn 1.9+."""
    import numpy as np
    for attr in ('feature_names_in_',):
        if hasattr(estimador, attr):
            try:
                delattr(estimador, attr)
            except Exception:
                pass
    # Recorrer pasos de un Pipeline
    if hasattr(estimador, 'steps'):
        for _, step in estimador.steps:
            _limpiar_feature_names(step)
    # Recorrer transformers de un ColumnTransformer
    if hasattr(estimador, 'transformers_'):
        for _, t, _ in estimador.transformers_:
            _limpiar_feature_names(t)
    if hasattr(estimador, 'transformers'):
        for _, t, _ in estimador.transformers:
            _limpiar_feature_names(t)


def cargar_modelos():
    """Carga los pipelines joblib desde el directorio del proyecto"""
    import sys
    import preprocesamiento as _prep
    # Los modelos fueron guardados desde un notebook donde las clases estaban en __main__.
    # Inyectarlas en __main__ permite que joblib/pickle las encuentre al deserializar.
    _main = sys.modules.get('__main__')
    if _main:
        for _name in dir(_prep):
            if not _name.startswith('_') and not hasattr(_main, _name):
                setattr(_main, _name, getattr(_prep, _name))
    try:
        for segmento, config in SEGMENTOS.items():
            ruta_modelo = os.path.join(RUTA_MODELOS, config['modelo_archivo'])
            modelo = joblib.load(ruta_modelo)
            _limpiar_feature_names(modelo)
            MODELOS[segmento] = modelo
        print("OK: Modelos cargados correctamente")
    except Exception as e:
        print(f"ERROR al cargar modelos: {e}")

def cargar_datos():
    """Carga los CSVs para estadísticas (opcional - no bloquea la predicción)"""
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
            except Exception as e:
                print(f"ERROR leyendo {archivo}: {e}")
        else:
            print(f"AVISO: CSV no encontrado: {archivo}")

def _parse_date(s):
    """Parsea fecha sin usar pd.to_datetime (evita crash en API C de numpy)."""
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
    """Convierte una fila de Reservas_22_23 a las 20 features del pipeline."""
    # ADR en USD
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

    # Antelación — usando _parse_date (Python puro, sin numpy datetime API)
    llegada_dt = _parse_date(row.get('LLEGADA', ''))
    toma_dt    = _parse_date(row.get('FECHA_TOMA', ''))
    try:
        antelacion = max(0, (llegada_dt - toma_dt).days) if (llegada_dt and toma_dt) else 0
    except Exception:
        antelacion = 0

    # Temporada
    try:
        mes = llegada_dt.month if llegada_dt else 0
        temporada = TEMPORADA_MESES.get(mes, 'ALTA')
    except Exception:
        temporada = 'ALTA'

    # País agrupado y distancia
    pais_raw    = str(row.get('PAIS', '') or '').strip().upper()
    pais_agrup  = MAPA_PAIS_AGRUPADO.get(pais_raw, 'Otros')
    distancia   = DISTANCIA_POR_PAIS.get(pais_agrup, 'No Info')

    # PAX_TIPO
    pax = int(row.get('PAX', 2) or 2)
    pax_tipo = 'SINGLE' if pax == 1 else ('PAREJAS' if pax == 2 else 'FAMILIAS')

    # Grupo
    try:
        grupo_cod = 1 if float(row.get('ID_MULTIPLE', 0) or 0) > 0 else 0
    except Exception:
        grupo_cod = 0

    # Segmento del hotel
    hotel_id = int(row.get('ID_HOTEL', 0) or 0)
    segmento  = HOTEL_SIZE_MAPPING.get(hotel_id, 'GRANDE')
    global_mean = GLOBAL_MEANS[segmento]

    fuente = str(row.get('FUENTE_NEGOCIO', '') or '').strip()

    features = {
        'ANTELACION_DIAS': antelacion,
        'ADR':             adr,
        'NOCHES':          noches,
        'NENES':           int(row.get('NENES', 0) or 0),
        'BEBES':           int(row.get('BEBES', 0) or 0),
        'SEGMENTO_COD':           global_mean,
        'PAIS_AGRUPADO_COD':      global_mean,
        'HABITACION_LIMPIA_COD':  global_mean,
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


def _generar_explicacion(features, meta):
    """Genera lista de factores explicativos para la predicción (reglas de dominio)."""
    factores = []

    # 1. Antelación — el predictor más importante en cancelaciones hoteleras
    ant = features.get('ANTELACION_DIAS', 0)
    if ant > 180:
        factores.append({'factor': 'Antelación muy alta', 'valor': f'{ant} días',
            'impacto': 'alto', 'direccion': 'sube',
            'descripcion': 'Reservas con >180 días de antelación tienen el mayor riesgo de cancelación. '
                           'A mayor tiempo entre reserva y llegada, más probable que cambien los planes.'})
    elif ant > 90:
        factores.append({'factor': 'Antelación alta', 'valor': f'{ant} días',
            'impacto': 'medio', 'direccion': 'sube',
            'descripcion': f'Con {ant} días de antelación, existe tiempo suficiente para que el cliente reconsidering. '
                           'La antelación es el predictor con mayor peso en el modelo.'})
    elif ant > 30:
        factores.append({'factor': 'Antelación moderada', 'valor': f'{ant} días',
            'impacto': 'bajo', 'direccion': 'neutro',
            'descripcion': f'{ant} días de antelación es un rango estándar con riesgo moderado.'})
    else:
        factores.append({'factor': 'Baja antelación', 'valor': f'{ant} días',
            'impacto': 'bajo', 'direccion': 'baja',
            'descripcion': f'Solo {ant} días entre reserva y llegada. Las reservas de última hora '
                           'rara vez se cancelan — el compromiso es alto.'})

    # 2. Canal de distribución
    if features.get('FUENTE_NEGOCIO_E-COMMERCE', 0):
        factores.append({'factor': 'Canal Online / OTA', 'valor': 'E-Commerce',
            'impacto': 'alto', 'direccion': 'sube',
            'descripcion': 'Las reservas por OTA y canales online tienen la mayor tasa histórica de cancelación. '
                           'La facilidad de cancelación online y las políticas flexibles aumentan el riesgo.'})
    elif features.get('FUENTE_NEGOCIO_DIRECT SALES', 0):
        factores.append({'factor': 'Venta directa', 'valor': 'Direct Sales',
            'impacto': 'medio', 'direccion': 'baja',
            'descripcion': 'Los clientes que reservan directamente muestran mayor fidelidad. '
                           'La venta directa reduce el riesgo de cancelación respecto a OTAs.'})
    elif features.get('FUENTE_NEGOCIO_T.O. / T.A.', 0):
        factores.append({'factor': 'Tour Operador / Agencia', 'valor': 'T.O. / T.A.',
            'impacto': 'bajo', 'direccion': 'baja',
            'descripcion': 'Los paquetes de T.O. y agencias suelen tener penalizaciones estrictas, '
                           'lo que desincentiva la cancelación.'})
    else:
        factores.append({'factor': 'Canal corporativo / otros', 'valor': 'Corporate/Others',
            'impacto': 'bajo', 'direccion': 'neutro',
            'descripcion': 'Canal con comportamiento mixto según el tipo de cliente corporativo.'})

    # 3. ADR
    adr = features.get('ADR', 0)
    if adr > 600:
        factores.append({'factor': 'Tarifa muy elevada', 'valor': f'${adr:.0f}/noche',
            'impacto': 'medio', 'direccion': 'sube',
            'descripcion': f'Con un ADR de ${adr:.0f}, la reserva tiene un coste elevado. '
                           'Las reservas de alto valor son reconsi deradas con más frecuencia.'})
    elif adr > 300:
        factores.append({'factor': 'Tarifa alta', 'valor': f'${adr:.0f}/noche',
            'impacto': 'bajo', 'direccion': 'neutro',
            'descripcion': f'ADR de ${adr:.0f} está en el rango medio-alto. Impacto moderado en la predicción.'})
    elif adr < 80:
        factores.append({'factor': 'Tarifa muy baja', 'valor': f'${adr:.0f}/noche',
            'impacto': 'bajo', 'direccion': 'baja',
            'descripcion': f'ADR de ${adr:.0f} sugiere posiblemente una tarifa no reembolsable u oferta especial, '
                           'lo que reduce la probabilidad de cancelación.'})

    # 4. Distancia / Origen geográfico
    dist = meta.get('distancia', 'No Info')
    pais = meta.get('pais_agrup', 'Otros')
    if dist == 'Largo':
        factores.append({'factor': 'Viajero de larga distancia', 'valor': pais,
            'impacto': 'medio', 'direccion': 'sube',
            'descripcion': f'Los viajeros intercontinentales ({pais}) tienen mayor incertidumbre: '
                           'visados, vuelos de conexión, costes elevados → mayor probabilidad de cancelación.'})
    elif dist == 'Corto':
        factores.append({'factor': 'Viajero de origen cercano', 'valor': pais,
            'impacto': 'bajo', 'direccion': 'baja',
            'descripcion': f'El mercado de proximidad ({pais}) presenta menor incertidumbre logística '
                           'y menor riesgo de cancelación por causas externas.'})

    # 5. Temporada
    temp = meta.get('temporada', 'ALTA')
    if temp == 'BAJA':
        factores.append({'factor': 'Temporada baja', 'valor': 'Jul-Sep',
            'impacto': 'medio', 'direccion': 'sube',
            'descripcion': 'La temporada baja se asocia con mayor flexibilidad en las políticas de cancelación '
                           'y clientes menos comprometidos con las fechas.'})
    elif temp == 'ALTA':
        factores.append({'factor': 'Temporada alta', 'valor': 'Dic-May',
            'impacto': 'bajo', 'direccion': 'baja',
            'descripcion': 'En temporada alta la demanda es elevada y los clientes suelen mantener su reserva '
                           'para no perder la plaza.'})

    # 6. Tipo de viajero
    pax = meta.get('pax_tipo', 'PAREJAS')
    if pax == 'SINGLE':
        factores.append({'factor': 'Viajero individual', 'valor': '1 pax',
            'impacto': 'medio', 'direccion': 'sube',
            'descripcion': 'Los viajeros individuales tienen más flexibilidad y menos compromisos, '
                           'lo que facilita la cancelación.'})
    elif pax == 'FAMILIAS':
        factores.append({'factor': 'Reserva familiar', 'valor': '3+ pax',
            'impacto': 'bajo', 'direccion': 'baja',
            'descripcion': 'Las reservas familiares tienen menor probabilidad de cancelación: '
                           'mayor compromiso, planificación anticipada y coste de cancelación más alto.'})

    return factores


def cargar_datos_originales():
    """Carga muestra de Reservas_22_23.csv, transforma al pipeline y obtiene predicciones."""
    global DATOS_ORIG, DF_COMPLETO
    ruta = os.path.join(RUTA_DATOS, 'Reservas_22_23.csv')
    if not os.path.exists(ruta):
        print("AVISO: Reservas_22_23.csv no encontrado")
        return
    try:
        cols = ['ID_RESERVA', 'ID_HOTEL', 'LLEGADA', 'FECHA_TOMA', 'NOCHES',
                'PAX', 'ADULTOS', 'NENES', 'BEBES', 'PAIS', 'FUENTE_NEGOCIO',
                'TIPO', 'STATUS', 'ID_MULTIPLE', 'MONEDA',
                'VALHAB', 'VALPEN', 'VALSERV', 'VALFIJOS']
        df = pd.read_csv(ruta, sep=';', nrows=10000, usecols=cols,
                         on_bad_lines='skip', low_memory=False)
        print(f"OK: CSV original leído ({len(df)} filas)")
        df['ID_HOTEL'] = pd.to_numeric(df['ID_HOTEL'], errors='coerce').fillna(0).astype(int)
        df = df[df['ID_HOTEL'].isin(HOTEL_SIZE_MAPPING.keys())]
        DF_COMPLETO = df  # guardamos el df completo para analytics
        muestra = df.sample(min(300, len(df)), random_state=42).reset_index(drop=True)
        print(f"OK: Muestra seleccionada ({len(muestra)} filas)")

        import numpy as np

        def _predecir(modelo, features_dict):
            clf = modelo.named_steps['modelo']
            X20 = np.array([[float(features_dict.get(c, 0.0)) for c in COLUMNAS_PIPELINE]],
                           dtype=np.float64)
            return clf.predict_proba(X20)

        predicciones_activas = False
        for seg, modelo in MODELOS.items():
            try:
                _predecir(modelo, {c: 0.0 for c in COLUMNAS_PIPELINE})
                predicciones_activas = True
                print(f"OK: predict_proba activo — 20 features ({seg})")
                break
            except Exception as e_test:
                print(f"AVISO: predict_proba no disponible — {type(e_test).__name__}: {e_test}")
                break

        registros = []
        for _, row in muestra.iterrows():
            try:
                features, meta = _transformar_fila(row)
            except Exception:
                continue

            prob = None
            riesgo = None
            if predicciones_activas:
                try:
                    seg_key = meta['segmento']
                    if seg_key in MODELOS:
                        prob_arr = _predecir(MODELOS[seg_key], features)
                        prob = round(float(prob_arr[0][1]) * 100, 1)
                        riesgo = 'ALTO' if prob > 65 else ('MEDIO' if prob > 35 else 'BAJO')
                except Exception:
                    pass

            status_val = int(row.get('STATUS', 0) or 0)
            cancelada  = bool(MAPA_STATUS.get(status_val, 0))
            hotel_id   = int(row.get('ID_HOTEL', 0))
            hotel_nom  = HOTEL_NOMBRES.get(hotel_id, f'Hotel {hotel_id}')
            explicacion = _generar_explicacion(features, meta)

            registros.append({
                'id_reserva':  str(row.get('ID_RESERVA', '')),
                'hotel_id':    hotel_id,
                'hotel_nom':   hotel_nom,
                'segmento':    meta['segmento'],
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
                '_features':   features,
            })

        DATOS_ORIG = registros
        preds_ok = sum(1 for r in registros if r['prob_pred'] is not None)
        print(f"OK: {len(DATOS_ORIG)} reservas cargadas ({preds_ok} con predicción)")
    except Exception as e:
        import traceback
        print(f"ERROR al cargar datos originales: {e}")
        traceback.print_exc()

cargar_modelos()
cargar_datos()
cargar_datos_originales()
_cargar_reservas_simuladas()


def _agregar_estadisticas():
    """Agrega reservas e ingresos por mes y por hotel desde DF_COMPLETO."""
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
    return redirect(url_for('login'))

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
    6:   {'nom': 'Palladium Vallarta',        'loc': 'Puerto Vallarta, México',     'pais': '🇲🇽', 'estrellas': 5, 'precio': 180, 'tag': 'Todo incluido', 'grad': 'linear-gradient(160deg,#0d7377,#14a085)', 'desc': 'Resort tropical frente al Pacífico con playas de arena blanca y selva exuberante.'},
    9:   {'nom': 'Dominican Fiesta H&C',      'loc': 'Punta Cana, Rep. Dom.',       'pais': '🇩🇴', 'estrellas': 4, 'precio': 130, 'tag': 'Playa privada', 'grad': 'linear-gradient(160deg,#c0392b,#8e44ad)', 'desc': 'Hotel & Casino frente al mar Caribe con entretenimiento para toda la familia.'},
    15:  {'nom': 'Ushuaïa Ibiza Beach Hotel', 'loc': 'Ibiza, España',               'pais': '🇪🇸', 'estrellas': 5, 'precio': 280, 'tag': 'Adults only', 'grad': 'linear-gradient(160deg,#6c3483,#d35400)', 'desc': 'Hotel de fiesta con los mejores DJs del mundo y vistas al Mar Mediterráneo.'},
    30:  {'nom': 'Hard Rock Hotel Ibiza',     'loc': 'Ibiza, España',               'pais': '🇪🇸', 'estrellas': 5, 'precio': 250, 'tag': 'Rock & Beach', 'grad': 'linear-gradient(160deg,#1a1a2e,#e74c3c)', 'desc': 'El espíritu del rock en el paraíso. Piscinas, spa y la mejor música en vivo.'},
    32:  {'nom': 'Hard Rock Hotel Tenerife',  'loc': 'Tenerife, España',            'pais': '🇪🇸', 'estrellas': 5, 'precio': 200, 'tag': 'Ocean view', 'grad': 'linear-gradient(160deg,#2d3436,#e17055)', 'desc': 'Impresionantes vistas al Atlántico desde las laderas del Teide.'},
    83:  {'nom': 'Grand Palladium Imbassai',  'loc': 'Bahia, Brasil',               'pais': '🇧🇷', 'estrellas': 5, 'precio': 160, 'tag': 'Todo incluido', 'grad': 'linear-gradient(160deg,#00b09b,#1e3c72)', 'desc': 'Naturaleza intacta entre la selva atlántica y las aguas cristalinas del nordeste brasileño.'},
    92:  {'nom': 'Grand Palladium Jamaica',   'loc': 'Montego Bay, Jamaica',        'pais': '🇯🇲', 'estrellas': 5, 'precio': 190, 'tag': 'Adults only', 'grad': 'linear-gradient(160deg,#1a6b3c,#f9ca24)', 'desc': 'El ritmo caribeño bajo las palmeras de la bahía más famosa de Jamaica.'},
    94:  {'nom': 'Grand Palladium Palace',    'loc': 'Ibiza, España',               'pais': '🇪🇸', 'estrellas': 5, 'precio': 310, 'tag': 'Luxury', 'grad': 'linear-gradient(160deg,#2c3e50,#3498db)', 'desc': 'Lujo mediterráneo en primera línea de playa con el exclusivo club de socios TRS.'},
    96:  {'nom': 'TRS Coral Hotel',           'loc': 'Costa Mujeres, México',       'pais': '🇲🇽', 'estrellas': 5, 'precio': 320, 'tag': 'Luxury adults', 'grad': 'linear-gradient(160deg,#0099f7,#00d2d3)', 'desc': 'Exclusivo resort adults-only frente a aguas turquesas a minutos de Cancún.'},
    99:  {'nom': 'TRS Cap Cana Hotel',        'loc': 'Cap Cana, Rep. Dom.',         'pais': '🇩🇴', 'estrellas': 5, 'precio': 350, 'tag': 'Ultra luxury', 'grad': 'linear-gradient(160deg,#0f0c29,#302b63)', 'desc': 'La joya del Caribe. Diseño arquitectónico único con butler service y gastronomía de autor.'},
    106: {'nom': 'Grand Palladium Riviera',   'loc': 'Playa del Carmen, México',    'pais': '🇲🇽', 'estrellas': 5, 'precio': 240, 'tag': 'Todo incluido', 'grad': 'linear-gradient(160deg,#11998e,#38ef7d)', 'desc': 'Selva maya, cenotes y playas turquesas en la Riviera Maya.'},
    107: {'nom': 'Grand Palladium Punta Cana','loc': 'Punta Cana, Rep. Dom.',       'pais': '🇩🇴', 'estrellas': 5, 'precio': 170, 'tag': 'Todo incluido', 'grad': 'linear-gradient(160deg,#005c97,#363795)', 'desc': 'Resort de playa con 11 piscinas, 14 restaurantes y el mejor todo incluido del Caribe.'},
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
    pais_raw = str(data.get('pais', '') or '').strip().upper()

    mes       = llegada_dt.month if llegada_dt else 6
    temporada = TEMPORADA_MESES.get(mes, 'ALTA')
    pais_agrup = MAPA_PAIS_AGRUPADO.get(pais_raw, 'Otros')
    distancia  = DISTANCIA_POR_PAIS.get(pais_agrup, 'No Info')
    pax        = adultos + nenes
    pax_tipo   = 'SINGLE' if pax == 1 else ('PAREJAS' if pax == 2 else 'FAMILIAS')
    global_mean = GLOBAL_MEANS.get(segmento, 0.35)

    features = {
        'ANTELACION_DIAS':              antelacion,
        'ADR':                          adr,
        'NOCHES':                       noches,
        'NENES':                        nenes,
        'BEBES':                        bebes,
        'SEGMENTO_COD':                 global_mean,
        'PAIS_AGRUPADO_COD':            global_mean,
        'HABITACION_LIMPIA_COD':        global_mean,
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
    riesgo = 'ALTO' if prob > 65 else ('MEDIO' if prob > 35 else 'BAJO')

    total_precio = round(adr * noches * adultos, 2)

    # Guardar en historial como reserva de canal cliente
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
        'explicacion':  _generar_explicacion(features, meta_h),
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

    # Mostrar: propias del usuario + las del canal público (reservas de clientes)
    sims_usuario = [r for r in RESERVAS_SIMULADAS
                    if r.get('usuario') == usuario or r.get('tipo') == 'cliente']
    sims_usuario = sorted(sims_usuario, key=lambda x: x.get('id',''), reverse=True)

    if not DATOS:
        return render_template('estadisticas.html', usuario=usuario,
                               totales=None, por_segmento={}, reservas=[],
                               datos_orig=datos_orig_full, segmentos_info=segmentos_info,
                               hoteles_disponibles=hoteles_disponibles,
                               analytics=analytics,
                               reservas_simuladas=sims_usuario)

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
                           reservas_simuladas=sims_usuario)



# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/nueva-reserva')
@login_required
def nueva_reserva():
    usuario = session.get('usuario')
    hoteles = [{'id': hid, 'nombre': nom, 'segmento': HOTEL_SIZE_MAPPING.get(hid, 'GRANDE')}
               for hid, nom in sorted(HOTEL_NOMBRES.items(), key=lambda x: x[1])]
    paises  = sorted(MAPA_PAIS_AGRUPADO.keys())

    # ADR medio por hotel calculado desde los datos reales
    adr_por_hotel = {}
    for r in DATOS_ORIG:
        hid = r.get('hotel_id')
        adr = r.get('adr', 0)
        if hid and adr and adr > 0:
            if hid not in adr_por_hotel:
                adr_por_hotel[hid] = []
            adr_por_hotel[hid].append(adr)
    adr_medias = {hid: round(sum(v) / len(v), 2) for hid, v in adr_por_hotel.items()}

    return render_template('nueva_reserva.html', usuario=usuario,
                           hoteles=hoteles, paises=paises, adr_medias=adr_medias)


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

    # Noches: calculadas desde fechas (salida - llegada)
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
    fuente    = 'E-COMMERCE'  # siempre online desde este simulador
    pais_raw  = str(data.get('pais', '') or '').strip().upper()

    mes        = llegada_dt.month if llegada_dt else 0
    temporada  = TEMPORADA_MESES.get(mes, 'ALTA')
    pais_agrup = MAPA_PAIS_AGRUPADO.get(pais_raw, 'Otros')
    distancia  = DISTANCIA_POR_PAIS.get(pais_agrup, 'No Info')
    pax        = adultos + nenes
    pax_tipo   = 'SINGLE' if pax == 1 else ('PAREJAS' if pax == 2 else 'FAMILIAS')

    features = {
        'ANTELACION_DIAS':              antelacion,
        'ADR':                          adr,
        'NOCHES':                       noches,
        'NENES':                        nenes,
        'BEBES':                        bebes,
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
    riesgo  = 'ALTO' if prob > 65 else ('MEDIO' if prob > 35 else 'BAJO')
    explicacion = _generar_explicacion(features, meta)

    # Guardar en historial de simulaciones
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
    print(f"\n{'='*60}")
    print("APLICACIÓN PALLADIUM - PREDICTOR DE CANCELACIONES")
    print(f"{'='*60}")
    print(f"Accediendo a: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"Modelos cargados: {list(MODELOS.keys())}")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False)
