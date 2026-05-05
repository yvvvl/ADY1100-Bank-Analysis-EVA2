"""Carga del dataset Bank Marketing desde `data/raw/` con rutas relativas al proyecto."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_RAW_FILENAME = "bank-additional-full.csv"
CSV_SEPARATOR = ";"
NOTEBOOKS_DIR_NAME = "notebooks"
_DATA_RAW = Path("data") / "raw"
_DATA_PROCESSED = Path("data") / "processed"
_OUTPUTS = Path("outputs")


def get_project_root() -> Path:
    """Raíz del proyecto según si la ejecución parte desde `notebooks/` o desde la raíz."""
    cwd = Path.cwd()
    return cwd.parent if cwd.name == NOTEBOOKS_DIR_NAME else cwd


def get_raw_data_path(filename: str = DEFAULT_RAW_FILENAME) -> Path:
    """Ruta absoluta/lógica al CSV en `data/raw/`."""
    return get_project_root() / _DATA_RAW / filename


def get_processed_dir() -> Path:
    """Directorio para artefactos tabulares procesados."""
    return get_project_root() / _DATA_PROCESSED


def get_outputs_dir() -> Path:
    """Directorio para reportes y figuras."""
    return get_project_root() / _OUTPUTS


def load_bank_data(filename: str = DEFAULT_RAW_FILENAME) -> pd.DataFrame:
    """Lee el CSV del banking dataset usando el separador configurado."""
    path = get_raw_data_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo esperado en: {path}")
    return pd.read_csv(path, sep=CSV_SEPARATOR)
