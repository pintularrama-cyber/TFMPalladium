# 🏨 Visual Palladium — Predictor de Cancelaciones

Aplicación web (TFM) que predice la **probabilidad de cancelación** de reservas hoteleras
del Palladium Hotel Group mediante modelos de Machine Learning, con explicaciones
basadas en **SHAP**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![ML](https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost-orange)

---

## ¿Qué hace?

- **Predice cancelaciones** de reservas usando tres modelos segmentados por tamaño de hotel
  (pequeño → Random Forest, mediano y grande → XGBoost).
- **Explica cada predicción** con valores SHAP: qué variables empujan la reserva hacia la
  cancelación y cuánto pesa cada una.
- **Dashboard interactivo**: reservas pasadas (resultado conocido) vs futuras (solo predicción),
  gráficos por mes/hotel e ingresos estimados.
- **Umbrales de riesgo por segmento** (ALTO / MEDIO / BAJO) calibrados para cada tipo de hotel.

---

## Instalación rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/eriikvidal/tfm-palladium.git
cd tfm-palladium

# 2. Crear entorno virtual e instalar dependencias
python3 -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Arrancar
python app.py
```

Abre tu navegador en **http://localhost:5050**

> 📖 Guía detallada (Mac/Windows + despliegue online) en [INSTRUCCIONES.md](INSTRUCCIONES.md)

---

## Credenciales de acceso

| Usuario   | Contraseña   |
|-----------|--------------|
| diego     | diego123     |
| erik      | erik123      |
| guillermo | guillermo123 |
| raul      | raul123      |

---

## Estructura

```
├── app.py               # Aplicación Flask + lógica de predicción y SHAP
├── config.py            # Configuración: segmentos, umbrales, mapeos
├── preprocesamiento.py  # Clases del pipeline de ML
├── modelos/             # Modelos entrenados (.joblib)
├── templates/           # Interfaz web (HTML)
├── entrenamiento/       # Notebook y datasets de entrenamiento
└── documentacion/       # Documentación técnica del TFM
```

> ⚠️ Los datasets originales completos (`Reservas_22_23.csv`, `df_grande.csv`) no se incluyen
> por su tamaño (>100 MB). La aplicación funciona con los modelos ya entrenados.

---

## Stack técnico

- **Backend**: Flask
- **ML**: scikit-learn (Random Forest), XGBoost
- **Explicabilidad**: SHAP (SHapley Additive exPlanations)
- **Datos**: pandas, numpy
- **Frontend**: HTML + Chart.js
