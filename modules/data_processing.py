# -*- coding: utf-8 -*-
"""
data_processing.py
Carga y procesamiento de datos. Mantiene EXACTAMENTE la misma lógica
del notebook original de Colab (sin modificar cálculos).
"""

import re
import numpy as np
import pandas as pd


FEATURES = [
    "Variabilidad",
    "Simetría",
    "Curvatura",
    "STDEV",
    "M5",
    "M1",
    "M3",
    "M4",
    "Simetría L3",
]

TARGET = "Reclamado"


def cargar_excel(archivo, sheet_name: str = "Dureza") -> pd.DataFrame:
    """Lee el Excel y limpia espacios en los nombres de columnas."""
    df = pd.read_excel(archivo, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    return df


def preparar_dataset(df: pd.DataFrame):
    """Selecciona features + target, elimina NaN y devuelve X, y, df_model."""
    df_model = df[FEATURES + [TARGET]].dropna().copy()
    X = df_model[FEATURES]
    y = df_model[TARGET]
    return X, y, df_model


def parsear_durezas(entrada: str) -> np.ndarray:
    """
    Convierte una entrada de texto con 20 valores (separados por
    espacios, saltos de línea, punto y coma) en un array float.
    Acepta coma decimal.
    """
    valores = re.split(r"[\n;\s]+", entrada.strip())
    durezas = [float(x.replace(",", ".")) for x in valores if x != ""]
    if len(durezas) != 20:
        raise ValueError(
            f"Debes ingresar 20 valores y recibiste {len(durezas)}"
        )
    return np.array(durezas)


def calcular_variables(arr: np.ndarray) -> dict:
    """
    Calcula las 9 variables usadas por el modelo a partir
    de las 20 mediciones de dureza. Idéntico al notebook.
    """
    M5 = float(np.mean(arr[0:4]))
    M4 = float(np.mean(arr[4:8]))
    M3 = float(np.mean(arr[8:12]))
    M1 = float(np.mean(arr[16:20]))

    Variabilidad = float(np.max(arr) - np.min(arr))
    STDEV = float(np.std(arr))
    Simetria = float(M5 - M1)

    Extremo = float(np.mean(np.concatenate([arr[1:4], arr[16:19]])))
    Curvatura = float(Extremo - M3)

    Simetria_L3 = float(M5 - M4)

    return {
        "Variabilidad": Variabilidad,
        "Simetría": Simetria,
        "Curvatura": Curvatura,
        "STDEV": STDEV,
        "M5": M5,
        "M1": M1,
        "M3": M3,
        "M4": M4,
        "Simetría L3": Simetria_L3,
    }


def variables_a_dataframe(variables: dict) -> pd.DataFrame:
    """Construye el DataFrame de 1 fila para predict_proba."""
    return pd.DataFrame([{k: variables[k] for k in FEATURES}])
