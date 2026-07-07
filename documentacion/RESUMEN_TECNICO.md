# Resumen Técnico — Palladium Cancellation Predictor

## ¿Qué hace esta aplicación?

Aplicación web Flask que predice la probabilidad de cancelación de reservas hoteleras, segmentada en tres tipos de hotel. También muestra un dashboard histórico con estadísticas reales de cancelaciones.

---

## Estructura del Proyecto

```
Proyecto Visual - Palladium/
│
├── app.py                     # Servidor Flask principal (rutas, lógica)
├── config.py                  # Configuración centralizada (rutas, constantes)
├── preprocesamiento.py        # Stubs necesarios para deserializar joblib
│
├── pipeline_pequeno.joblib    # Modelo entrenado — Hotel PEQUEÑO (<250 hab)
├── pipeline_mediano.joblib    # Modelo entrenado — Hotel MEDIANO (250–500 hab)
├── pipeline_grande.joblib     # Modelo entrenado — Hotel GRANDE (>500 hab)
│
├── df_pequeno.csv             # Datos históricos PEQUEÑO (27.113 reservas)
├── df_mediano.csv             # Datos históricos MEDIANO (152.768 reservas)
├── df_grande.csv              # Datos históricos GRANDE (965.010 reservas)
│
├── templates/
│   ├── base.html              # Navbar + footer compartido
│   ├── login.html             # Pantalla de login
│   ├── dashboard.html         # Página de inicio tras login
│   ├── predictor.html         # Formulario de predicción (usuarios internos)
│   ├── estadisticas.html      # Dashboard histórico con filtros
│   ├── index.html             # Landing page pública
│   └── cliente_reserva_simple.html  # Formulario público para clientes
│
└── venv/                      # Entorno virtual Python
```

---

## Los tres Modelos ML

Hay un modelo por segmento de hotel, seleccionado automáticamente según el número de habitaciones:

| Segmento | Habitaciones | Algoritmo     | AUC-ROC |
|----------|-------------|---------------|---------|
| PEQUEÑO  | < 250       | Random Forest | 0.8425  |
| MEDIANO  | 250–500     | XGBoost       | 0.8425  |
| GRANDE   | > 500       | XGBoost       | 0.8169  |

### Estructura interna de cada pipeline
Cada `.joblib` es un `sklearn.pipeline.Pipeline` con estos pasos:

```
1. target_enc  → TargetEncoderSmoothing  (no-op — datos ya vienen codificados)
2. binary_enc  → BinaryEncoderCustom     (no-op — datos ya vienen codificados)
3. preprocessor → ColumnTransformer:
       - StandardScaler sobre 6 columnas numéricas
       - Passthrough sobre las 14 columnas restantes (COD + OHE)
4. classifier  → XGBClassifier / RandomForestClassifier
```

### Por qué `preprocesamiento.py` es imprescindible
Las clases `TargetEncoderSmoothing` y `BinaryEncoderCustom` no son de sklearn estándar. Al hacer `joblib.load()`, Python necesita que esas clases existan en memoria o lanza `ModuleNotFoundError`. El archivo `preprocesamiento.py` contiene stubs (esqueletos vacíos) que satisfacen esa dependencia sin necesidad de reimplementar la lógica real.

---

## Las 20 columnas del modelo

El ColumnTransformer espera exactamente estas columnas en este orden:

```python
COLUMNAS_PIPELINE = [
    # Numéricas (van a StandardScaler)
    'ANTELACION_DIAS', 'ADR', 'NOCHES', 'NENES', 'BEBES',
    # Target-encoded (float entre 0 y 1, aprox. tasa de cancelación del grupo)
    'SEGMENTO_COD', 'PAIS_AGRUPADO_COD', 'HABITACION_LIMPIA_COD',
    # Binaria
    'GRUPO_TIPO_COD',
    # OHE Fuente de Negocio (baseline = Corporate)
    'FUENTE_NEGOCIO_DIRECT SALES', 'FUENTE_NEGOCIO_E-COMMERCE',
    'FUENTE_NEGOCIO_OTHERS', 'FUENTE_NEGOCIO_T.O. / T.A.',
    # OHE Tipo de Pax (baseline = Familias)
    'PAX_TIPO_PAREJAS', 'PAX_TIPO_SINGLE',
    # OHE Temporada (baseline = Alta)
    'TEMPORADA_BAJA', 'TEMPORADA_MEDIA',
    # OHE Distancia (baseline = Corto)
    'DISTANCIA_LARGO', 'DISTANCIA_MEDIO', 'DISTANCIA_NO INFO'
]
```

**Importante:** los CSVs ya tienen los datos en este formato pre-codificado. El predictor construye este DataFrame manualmente a partir de los inputs del usuario.

**Limitación conocida:** `SEGMENTO_COD`, `PAIS_AGRUPADO_COD` y `HABITACION_LIMPIA_COD` son valores target-encoded cuyo mapeo original (string → float) no se guardó en el pipeline. Para la predicción se usa el `global_mean` del segmento como aproximación. Las otras 17 columnas se calculan correctamente.

---

## Cómo funciona el Predictor

El usuario rellena el formulario con datos reales de la reserva:

```
Inputs del usuario          →   Feature del modelo
──────────────────────────────────────────────────
Fecha checkin/checkout      →   NOCHES, ANTELACION_DIAS, TEMPORADA_*
Adultos/niños/bebés         →   NENES, BEBES, PAX_TIPO_*, GRUPO_TIPO_COD
Nº habitaciones hotel       →   Selección del modelo (PEQUEÑO/MEDIANO/GRANDE)
ADR (tarifa)                →   ADR
Fuente de negocio           →   FUENTE_NEGOCIO_*
País de origen              →   DISTANCIA_* (proxy geográfico)
```

Las columnas COD usan `GLOBAL_MEANS` por segmento:
- PEQUEÑO: 0.487 · MEDIANO: 0.284 · GRANDE: 0.396

---

## El Dashboard de Estadísticas

La página de Estadísticas lee los CSVs históricos directamente y muestra datos reales. **No ejecuta el modelo de ML** — simplemente agrega `STATUS_BOOL` (resultado real de cancelación).

- KPIs globales: total reservas, cancelaciones reales, tasa
- Cards por segmento con tasas reales e info del modelo
- Gráfico de barras apiladas (Chart.js)
- Tabla con 600 reservas muestreadas (200 por segmento) con filtros

Las columnas OHE de los CSVs se reconstruyen para mostrar valores legibles:
- `FUENTE_NEGOCIO_E-COMMERCE == 1` → "E-COMMERCE", si no → baseline "Corporate"
- `TEMPORADA_BAJA == 1` → "Baja", etc.

---

## Datos del dataset histórico

| Segmento | Reservas  | Canceladas | Tasa   |
|----------|-----------|------------|--------|
| PEQUEÑO  | 27.113    | 13.212     | 48.7%  |
| MEDIANO  | 152.768   | 43.329     | 28.4%  |
| GRANDE   | 965.010   | 381.716    | 39.6%  |
| **TOTAL**| **1.144.891** | **438.257** | **38.3%** |

---

## Rutas del servidor Flask

| Ruta                  | Acceso   | Descripción                       |
|-----------------------|----------|-----------------------------------|
| `/`                   | Público  | Landing page                      |
| `/cliente/reserva`    | Público  | Formulario de reserva para cliente|
| `/cliente/predecir`   | Público  | POST — predicción desde cliente   |
| `/login`              | Público  | Login interno                     |
| `/dashboard`          | Login    | Página de inicio de usuarios      |
| `/predictor`          | Login    | Predictor interno (formulario)    |
| `/estadisticas`       | Login    | Dashboard histórico               |
| `/logout`             | Login    | Cerrar sesión                     |

---

## Usuarios del sistema

```
diego / diego123
erik / erik123
guillermo / guillermo123
raul / raul123
```
*(definidos en `config.py → USUARIOS`)*

---

## Cambiar rutas (Mac ↔ Windows)

En `config.py`, línea 18:
```python
# Por defecto: usa la carpeta del proyecto automáticamente
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mac manual (descomenta y ajusta):
# BASE_DIR = '/Users/erik/Library/Mobile Documents/...'

# Windows manual (descomenta y ajusta):
# BASE_DIR = r'C:\Users\mejora2\Desktop\Pruebas TFM\Proyecto Visual - Palladium'
```

---

## Dependencias principales

```
flask==3.0.3
joblib
scikit-learn==1.9.0    (modelos entrenados con 1.6.1 — warnings no bloqueantes)
xgboost==2.0.3
pandas
numpy
```

Instalar: `.\venv\Scripts\pip.exe install flask joblib scikit-learn xgboost pandas`

Arrancar: `.\venv\Scripts\python.exe app.py`

Acceder: `http://localhost:5001`
