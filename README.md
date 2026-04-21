# Predicción de Bagginess — Testliner

Interfaz Streamlit que reutiliza **la misma lógica** del notebook original
de Google Colab (sin modificar cálculos ni el modelo XGBoost).

## Estructura

```
app/
├── app.py                      # Interfaz Streamlit (frontend)
├── requirements.txt
├── modules/                    # Backend modularizado
│   ├── data_processing.py      # Carga Excel y cálculo de variables
│   ├── model_training.py       # Split, XGBoost, reglas híbridas
│   ├── prediction.py           # Predicción de un rollo
│   ├── visualizations.py       # Gráficos matplotlib
│   └── audit.py                # Logging estructurado (auditoría)
├── models/                     # Modelos entrenados (.joblib)
├── logs/                       # Auditoría (auditoria.log + auditoria.jsonl)
└── data/                       # Archivos de ejemplo
```

## Ejecutar localmente

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

La UI abrirá en `http://localhost:8501`.

## Flujo de uso

1. **Carga de datos** — Subir el Excel (hoja `Dureza` por defecto).
2. **Entrenamiento** — Botón "Entrenar modelo" (muestra logs y guarda `models/xgb_model.joblib`).
3. **Resultados** — Matriz de confusión, reporte, importancia de variables, distribuciones.
4. **Predicción de rollo** — 20 inputs de dureza → probabilidad + clasificación (🔴 / 🟡 / 🟢).
5. **Auditoría** — Todos los eventos (carga, entrenamiento, predicciones, comparaciones) con timestamp.
6. **Comparación con Colab** — Sube un CSV con `Prob_modelo`, `Decision`, `Pred_final` y compara fila a fila.

## Consistencia con Colab

- Mismos features (9) y target (`Reclamado`).
- Mismo split: `test_size=0.25`, `random_state=42`, `stratify=y`.
- Mismos hiperparámetros XGBoost: `n_estimators=400`, `max_depth=4`,
  `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`,
  `scale_pos_weight` calculado igual.
- Reglas híbridas idénticas (umbrales 0.75 / 0.35+STDEV>3.3 / 0.30 o STDEV>3).
- Cálculo de variables desde 20 durezas idéntico (M5, M4, M3, M1, Simetría, Curvatura, etc.).

## Archivo de entrada esperado

Excel `.xlsx` con hoja `Dureza` (configurable) que contenga como mínimo las columnas:

```
Variabilidad, Simetría, Curvatura, STDEV, M5, M1, M3, M4, Simetría L3, Reclamado
```

## Auditoría

Cada acción queda registrada en:
- `logs/auditoria.log` (texto)
- `logs/auditoria.jsonl` (una línea JSON por evento, descargable desde la UI)
