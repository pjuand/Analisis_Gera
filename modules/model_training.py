# -*- coding: utf-8 -*-
"""
model_training.py
Entrenamiento del modelo XGBoost. Mantiene los mismos hiperparámetros
que el notebook original de Colab.
"""

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


MODEL_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "xgb_model.joblib",
)


def split_datos(X: pd.DataFrame, y: pd.Series):
    """División 75/25 estratificada, random_state=42 (idéntico a Colab)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def entrenar_modelo(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Entrena XGBoost con exactamente los mismos parámetros del notebook."""
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    return xgb


def evaluar_modelo(xgb: XGBClassifier, X: pd.DataFrame, y: pd.Series,
                   df_model: pd.DataFrame, target: str = "Reclamado") -> dict:
    """
    Aplica la clasificación híbrida (modelo + reglas físicas) idéntica
    al notebook y devuelve matriz de confusión + reporte.
    """
    df_scores = df_model.copy()
    df_scores["Prob_modelo"] = xgb.predict_proba(X)[:, 1]

    df_scores["Regla_STDEV"] = df_scores["STDEV"] > 3.2
    df_scores["Regla_Curvatura"] = df_scores["Curvatura"] > 4
    df_scores["Regla_Variabilidad"] = df_scores["Variabilidad"] > 10
    df_scores["Regla_Fisica"] = (
        df_scores["Regla_STDEV"]
        | df_scores["Regla_Curvatura"]
        | df_scores["Regla_Variabilidad"]
    )

    df_scores["Decision"] = df_scores.apply(_clasificar, axis=1)
    df_scores["Pred_final"] = (df_scores["Decision"] == "🔴 RECHAZAR").astype(int)

    cm = confusion_matrix(df_scores[target], df_scores["Pred_final"])
    report = classification_report(
        df_scores[target], df_scores["Pred_final"], output_dict=True
    )
    report_text = classification_report(df_scores[target], df_scores["Pred_final"])

    importancias = pd.DataFrame({
        "Variable": X.columns.tolist(),
        "Importancia": xgb.feature_importances_,
    }).sort_values(by="Importancia", ascending=False).reset_index(drop=True)

    return {
        "df_scores": df_scores,
        "confusion_matrix": cm,
        "classification_report": report,
        "classification_report_text": report_text,
        "importancias": importancias,
        "prob_stats": {
            "min": float(df_scores["Prob_modelo"].min()),
            "max": float(df_scores["Prob_modelo"].max()),
            "std": float(df_scores["Prob_modelo"].std()),
        },
        "distribucion_decisiones": df_scores["Decision"].value_counts().to_dict(),
    }


def _clasificar(row) -> str:
    """Regla híbrida idéntica a la usada en Colab (celda 7)."""
    if row["Prob_modelo"] >= 0.75:
        return "🔴 RECHAZAR"
    elif row["Prob_modelo"] >= 0.35 and row["STDEV"] > 3.3:
        return "🔴 RECHAZAR"
    elif row["Prob_modelo"] >= 0.3 or row["STDEV"] > 3:
        return "🟡 ALERTA"
    else:
        return "🟢 OK"


def guardar_modelo(xgb: XGBClassifier, path: str = MODEL_PATH_DEFAULT) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(xgb, path)
    return path


def cargar_modelo(path: str = MODEL_PATH_DEFAULT) -> XGBClassifier:
    return joblib.load(path)
