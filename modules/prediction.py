# -*- coding: utf-8 -*-
"""
prediction.py
Predicción para un rollo individual a partir de 20 mediciones de dureza.
Reproduce la lógica de decisión del notebook (celdas finales).
"""

import pandas as pd
from xgboost import XGBClassifier

from .data_processing import (
    calcular_variables,
    variables_a_dataframe,
    parsear_durezas,
)


def decidir(prob: float, STDEV: float) -> str:
    """Regla de decisión individual IDÉNTICA al notebook (celda final)."""
    if prob >= 0.75:
        return "🔴 RECHAZAR"
    elif prob >= 0.30 or STDEV > 3:
        return "🟡 ALERTA"
    else:
        return "🟢 OK"


def predecir_rollo(xgb: XGBClassifier, arr) -> dict:
    """
    Pipeline completo para un rollo:
      arr (20 valores) -> variables -> predict_proba -> decisión
    """
    variables = calcular_variables(arr)
    X = variables_a_dataframe(variables)
    prob = float(xgb.predict_proba(X)[:, 1][0])
    decision = decidir(prob, variables["STDEV"])

    return {
        "durezas": list(map(float, arr)),
        "variables": variables,
        "X": X,
        "probabilidad": prob,
        "decision": decision,
    }


def predecir_desde_texto(xgb: XGBClassifier, entrada: str) -> dict:
    """Versión para entrada de texto (20 valores separados)."""
    arr = parsear_durezas(entrada)
    return predecir_rollo(xgb, arr)
