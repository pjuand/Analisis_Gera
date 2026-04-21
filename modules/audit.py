# -*- coding: utf-8 -*-
"""
audit.py
Sistema de auditoría: logging estructurado a archivo + memoria.
Cada evento guarda fecha, tipo, datos de entrada, variables calculadas y salida.
"""

import os
import json
import logging
from datetime import datetime


LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
)
os.makedirs(LOG_DIR, exist_ok=True)

_LOG_FILE = os.path.join(LOG_DIR, "auditoria.log")
_JSONL_FILE = os.path.join(LOG_DIR, "auditoria.jsonl")

logger = logging.getLogger("bagginess_audit")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)


def registrar_evento(tipo: str, payload: dict) -> dict:
    """Registra un evento de auditoría y lo devuelve con timestamp."""
    evento = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tipo": tipo,
        "payload": _jsonable(payload),
    }
    logger.info(f"{tipo} | {json.dumps(evento['payload'], ensure_ascii=False)}")
    with open(_JSONL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")
    return evento


def leer_eventos(limit: int = 200) -> list:
    if not os.path.exists(_JSONL_FILE):
        return []
    with open(_JSONL_FILE, "r", encoding="utf-8") as f:
        lineas = f.readlines()
    eventos = []
    for linea in lineas[-limit:]:
        try:
            eventos.append(json.loads(linea))
        except Exception:
            continue
    return eventos


def _jsonable(obj):
    """Convierte numpy / pandas a tipos serializables."""
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj
