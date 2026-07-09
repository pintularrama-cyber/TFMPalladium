# Estado del proyecto — Visual Palladium (TFM)

> Registro del estado actual y del trabajo realizado. Última actualización: julio 2026.
> Repositorio: https://github.com/eriikvidal/tfm-palladium (privado)

---

## Qué es

Aplicación web (Flask) que predice la probabilidad de cancelación de reservas hoteleras
del Palladium Hotel Group, con explicaciones basadas en SHAP.

## Funcionalidades implementadas

### Predicción y explicabilidad
- **Modelos por segmento**: Random Forest (pequeño), XGBoost (mediano y grande).
- **AUC reales optimizados**: 0.8473 / 0.8668 / 0.8513 (verificados en los notebooks).
- **SHAP** (TreeExplainer): cada predicción se explica con la contribución real de cada variable.
- **Codificación real** (Target Encoding, `mapeos_encoding.json`): país y segmento de mercado
  usan su tasa histórica real; la habitación usa media global (depende del hotel).

### Interfaz
- **Página de inicio (clientes)**: 8 hoteles reales con **foto real** cada uno + simulador de reserva.
- **Login** de empleados (usuarios en config.py).
- **Análisis (dashboard)**:
  - Gráficos por mes: pasadas (activas/canceladas) vs futuras (riesgo alto/medio/bajo).
  - Ingresos: ganados / perdidos (pasado) vs en riesgo por nivel (futuro).
  - Drill-down por mes y por hotel (modal con completadas + futuras).
  - Fecha de corte compartida y reactiva (AJAX).
- **Histórico**: tablas Pasadas / Futuras con filtros (hotel, tamaño, estado, riesgo, fuente, temporada).
  - En futuras NO se muestra el resultado real (solo predicción); en pasadas sí.
- **Reservas**: historial de reservas generadas, con explicación SHAP.

### Datos — todo verificado, nada inventado
- 8 hoteles reales de América (catálogo real); se eliminaron hoteles de España inventados.
- Habitaciones, AUC, distancia, país: todo cruzado con los datos/notebooks reales.
- Ver **AUDITORIA_DATOS.md** para la trazabilidad completa y las frases de defensa ante el tribunal.

## Cómo arrancar (local)

```bash
source venv/bin/activate      # Windows: venv\Scripts\activate
python app.py
```
Abrir http://localhost:5050 · Login: erik / erik123 (u otros en config.py)

## Despliegue web (Render)
- `app.run` usa host `0.0.0.0` y puerto de la variable `PORT`.
- Start Command sugerido: `gunicorn app:app --bind 0.0.0.0:$PORT`
- Nota: los CSV grandes (`Reservas_22_23.csv`, `df_grande.csv`) no están en el repo, así que
  en Render la tabla de reservas reales saldría vacía (las predicciones en vivo sí funcionan).

## Archivos que NO están en el repo (grandes / privados)
- `venv/`, `datos/Reservas_22_23.csv`, `entrenamiento/df_grande.csv`, `drive documentos/`
  (notebooks de entrenamiento y datasets originales, 1 GB, solo en local).

## Scripts de verificación
- `verificar_shap.py` — comprueba que SHAP funciona con los modelos.
- `verificar_auc.py` — recalcula el AUC real sobre los datos.

## Pendiente / a revisar
- Revisar las fotos de los hoteles (comprobar que cada foto corresponde al hotel correcto;
  en concreto confirmar la del TRS Cap Cana vs Punta Cana).
