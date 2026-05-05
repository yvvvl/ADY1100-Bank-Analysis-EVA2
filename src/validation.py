"""Validación de estructura y calidad básica del dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict

import pandas as pd

UNKNOWN_TOKEN = "unknown"
_READ_CHUNK_BYTES = 8192

EXPECTED_COLUMNS: tuple[str, ...] = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
)


class ColumnValidationResult(TypedDict):
    missing_columns: list[str]
    extra_columns: list[str]
    is_valid: bool


def validate_columns(df: pd.DataFrame) -> ColumnValidationResult:
    """Contrasta columnas esperadas frente a las presentes en el DataFrame."""
    expected = set(EXPECTED_COLUMNS)
    actual = set(df.columns)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return {
        "missing_columns": missing,
        "extra_columns": extra,
        "is_valid": len(missing) == 0,
    }


def get_basic_summary(df: pd.DataFrame) -> dict:
    """Resumen de forma, tipos, duplicados y nulos (una sola lectura pasiva del estado del dataset)."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "duplicated_rows": int(df.duplicated().sum()),
        "null_values": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def count_unknown_values(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega conteos de la cadena configurada como desconocido por columna."""
    if df.empty:
        return pd.DataFrame(columns=["unknown_count", "unknown_percent"])
    mask_unknown = df.astype(object).eq(UNKNOWN_TOKEN)
    unknown_counts = mask_unknown.sum()
    unknown_percent = (unknown_counts / len(df)) * 100
    positive = unknown_counts[unknown_counts > 0]
    return (
        positive.to_frame("unknown_count").assign(
            unknown_percent=unknown_percent[positive.index]
        )
    )


def compute_md5_checksum(file_path: Path | str) -> str:
    """Hash MD5 del archivo para comprobar integridad del raw sin cargarlo en memoria."""
    path = Path(file_path)
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calcular_checksum_md5(file_path: Path | str) -> str:
    """Alias en español para mantener compatibilidad con notebooks existentes."""
    return compute_md5_checksum(file_path)
