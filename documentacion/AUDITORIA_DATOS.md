# Auditoría de datos — Visual Palladium (TFM)

> Documento de trazabilidad: de dónde viene cada dato del proyecto y qué se verificó.
> Objetivo: garantizar que **ningún dato está inventado** y poder defenderlo ante el tribunal.
> Fecha de la auditoría: julio 2026.

---

## 1. Predicciones

- **Cómo funciona**: cada reserva se convierte en 20 variables numéricas (`COLUMNAS_PIPELINE`)
  y el modelo entrenado (`.joblib`) calcula la probabilidad de cancelación con `predict_proba`.
- **Estado**: ✅ Real. La predicción es la salida directa del modelo.
- ⚠️ Matiz de vocabulario: **el modelo predice; SHAP explica**. No se "predice con SHAP".

## 2. Explicabilidad (SHAP)

- La app explica cada predicción con **SHAP (TreeExplainer)**: descompone el % en la
  contribución de cada variable.
- **Verificado**: tu notebook `Modelos finales (explicabilidad).ipynb` **ya usaba SHAP**
  con los mismos 3 modelos. El enfoque de la app es idéntico al de tu TFM.
- **Estado**: ✅ Real y verificado (`verificar_shap.py` lo confirma en ejecución).

## 3. Modelos y métricas (AUC-ROC)

Fuente: notebooks `modelos avanzados I (sin/con tunning)` y `Encapsulado Modelos finales`
(carpeta `drive documentos/colabs`).

| Segmento | Algoritmo | Filas entren. | AUC (sin tuning) | AUC (con tuning) → **usado** |
|----------|-----------|---------------|------------------|------------------------------|
| PEQUEÑO  | Random Forest | 27.113   | 0.8425 | **0.8473** |
| MEDIANO  | XGBoost       | 152.768  | 0.8425 | **0.8668** |
| GRANDE   | XGBoost       | 965.010  | 0.8169 | **0.8513** |

- Los AUC se midieron sobre el **conjunto de test** (`train_test_split`).
- El `.joblib` MEDIANO usa **exactamente** los hiperparámetros del GridSearch
  (`n_estimators=280, max_depth=11, learning_rate=0.08`), confirmando que son los optimizados.
- Hiperparámetros de producción (notebook Encapsulado):
  - PEQUEÑO RF: `n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=2, max_features='log2', class_weight='balanced'`
  - MEDIANO XGB: `n_estimators=280, max_depth=11, learning_rate=0.08, subsample=0.75, colsample_bytree=0.9, gamma=0.2, scale_pos_weight=ratio`
  - GRANDE XGB: `n_estimators=300, max_depth=12, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, gamma=0, scale_pos_weight=ratio`
- **Estado**: ✅ Real. Config actualizado a los AUC optimizados.

## 4. Hoteles

- **Datos reales**: 8 hoteles, **todos de América** (`catalogo_hoteles_final.csv` y el CSV de reservas).

| ID | Nombre | Segmento | ADR medio (USD) |
|----|--------|----------|-----------------|
| 6   | Palladium Vallarta | MEDIANO | 215,74 |
| 9   | Dominican Fiesta Hotel & Casino | MEDIANO | 100,17 |
| 83  | Grand Palladium Imbassai Resort & Spa | GRANDE | 392,05 |
| 92  | Grand Palladium Jamaica & Lady Hamilton | GRANDE | 235,36 |
| 96  | Complejo Costa Mujeres | GRANDE | 305,21 |
| 99  | TRS Cap Cana | PEQUEÑO | 267,32 |
| 106 | Complejo Riviera Maya | GRANDE | 241,71 |
| 107 | Complejo Punta Cana | GRANDE | 180,06 |

- **CORREGIDO**: se eliminaron 4 hoteles de **España inventados** (Ushuaïa Ibiza, Hard Rock
  Ibiza, Hard Rock Tenerife, GP Palace Ibiza) que no existen en ningún dato, y ~40 IDs de
  hotel falsos del mapeo de segmentos.
- Los precios de la web ahora son el **ADR medio real** de cada hotel.

## 5. Otras variables

| Variable | Estado |
|----------|--------|
| `DISTANCIA_POR_PAIS` (país→corto/medio/largo) | ✅ Real — variable del modelo (columnas DISTANCIA_* en los datos de entrenamiento) |
| Habitaciones | ✅ CORREGIDO — ahora las reales (Suite, Deluxe, Junior Suite, Superior, Presidential, Standard); se quitaron "Doble"/"Sencilla" inventadas |
| España como país de origen | ✅ Real — 38.906 reservas de clientes españoles |
| Mapeos de limpieza (status, temporada, país) | ✅ Real — notebook de limpieza |
| ADR medio por hotel | ✅ Real — `adr_medio_por_hotel.csv` |

## 6. Codificación de país/habitación/segmento (RESUELTO)

- En el entrenamiento, `SEGMENTO_COD`, `PAIS_AGRUPADO_COD` y `HABITACION_LIMPIA_COD` se
  calcularon con **Target Encoding suavizado** (smoothing=10): cada categoría → su tasa de
  cancelación suavizada. (Fuente: `FEATURE ENGINEERING.ipynb`.)
- El diccionario de mapeo no se serializó en el `.joblib`, así que originalmente la app usaba
  la media global como aproximación.
- **Solución aplicada**: se exportó el diccionario real desde el notebook FEATURE ENGINEERING
  al archivo **`mapeos_encoding.json`** (mismo `df`, misma función, mismo smoothing=10). Se
  verificó que coincide con los valores del entrenamiento (ej: ALEMANIA → 0.2219, idéntico al
  valor COD real). La app ahora lo carga y codifica cada categoría con su **valor real**.
- **Estado por variable:**
  - **País de origen**: valor real en toda la app (reservas reales y simulador).
  - **Segmento de mercado**: valor real en reservas reales (columna del CSV) y en el simulador
    (selector añadido al formulario).
  - **Tipo de habitación**: valor real en el simulador (selector añadido). En las reservas
    reales del CSV se usa la media global, porque `Reservas_22_23.csv` no incluye la habitación
    ya categorizada.
- **Ejemplos de valores reales** (tasa histórica de cancelación): España 0.283 · USA 0.424 ·
  Brasil 0.441 · Alemania 0.222 · GROUP LEISURE 0.214 · SUITE 0.397.

## 7. Decisión de diseño: qué variable se pide y cuál se aproxima

Regla aplicada: **una variable solo se pide al usuario si es fiable; si no, se usa el valor
medio** (aproximación neutral, nunca un dato inventado).

| Variable | ¿Depende del hotel? | En la app |
|----------|---------------------|-----------|
| **País de origen** | No (lo aporta el cliente) | ✅ Valor real (Target Encoding) |
| **Segmento de mercado** | No (BAR, Grupos, Bodas… cualquier hotel puede tenerlos) | ✅ Valor real (lo elige el empleado; en histórico viene del CSV) |
| **Tipo de habitación** | **Sí** (cada hotel tiene su oferta) | Media global — **no se pide** para no generar combinaciones imposibles |

**Por qué no se pide la habitación:** forzar un desplegable con todas las categorías (Villa,
Ambassador, Loft Suite…) permitiría elegir una habitación que ese hotel no tiene → dato
erróneo. Usar el valor medio es lo honesto: *"no dispongo de este dato concreto, uso el
promedio"*. No es inventar; inventar sería asignar una habitación imposible.

## 8. Cómo defenderlo ante el tribunal (frases listas)

- "Las predicciones salen de modelos Random Forest y XGBoost entrenados por segmento de hotel."
- "Las explicaciones usan SHAP, igual que en nuestro notebook de explicabilidad."
- "Los AUC (0.85–0.87) son los de los modelos optimizados, medidos sobre test."
- "Los 8 hoteles y sus datos provienen del catálogo y el dataset reales; no hay datos inventados."
- "El país y el segmento de mercado se codifican con su valor real (tasa histórica de
  cancelación). La habitación no se solicita porque cada hotel tiene una oferta distinta;
  forzar una categoría genérica introduciría ruido, así que se usa el valor medio del encoder."
- Si preguntan por qué algunas variables usan la media: "es una aproximación neutral cuando no
  disponemos del dato concreto; el modelo se entrenó y evaluó con los valores reales."

---

*Scripts de verificación incluidos: `verificar_shap.py` (comprueba SHAP) y `verificar_auc.py`
(calcula el AUC real sobre los datos).*
