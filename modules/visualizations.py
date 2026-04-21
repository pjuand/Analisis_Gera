# -*- coding: utf-8 -*-
"""
visualizations.py
Gráficos reutilizables (matplotlib) para la interfaz Streamlit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatter_prob_vs_var(df_scores: pd.DataFrame, variable: str, titulo: str = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df_scores[variable], df_scores["Prob_modelo"], alpha=0.7)
    ax.set_xlabel(variable)
    ax.set_ylabel("Probabilidad")
    ax.set_title(titulo or f"Riesgo vs {variable}")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_matriz_confusion(cm: np.ndarray, labels=("No reclamado", "Reclamado")):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_importancias(importancias: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importancias["Variable"][::-1], importancias["Importancia"][::-1])
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de variables (XGBoost)")
    fig.tight_layout()
    return fig


def plot_distribucion_variable(df: pd.DataFrame, variable: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[variable].dropna(), bins=25, edgecolor="black", alpha=0.8)
    ax.set_xlabel(variable)
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución: {variable}")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_perfil_rollo(arr):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(1, len(arr) + 1), arr, marker="o")
    ax.set_xlabel("Posición (1-20)")
    ax.set_ylabel("Dureza")
    ax.set_title("Perfil de dureza del rollo")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig
