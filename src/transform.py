"""Transformaciones: codificación de objetivo, one-hot y escalamiento."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN_DEFAULT = "y"
_TARGET_MAP: Final[dict[str, int]] = {"yes": 1, "no": 0}
_EXPECTED_LABELS = frozenset(_TARGET_MAP.keys())


def encode_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN_DEFAULT) -> pd.DataFrame:
    """Codifica etiquetas yes/no en enteros {1, 0}. Falla ante valores fuera del dominio."""
    out = df.copy()
    if target_col not in out.columns:
        return out
    col = out[target_col]
    if pd.api.types.is_bool_dtype(col):
        out[target_col] = col.astype(np.int8)
        return out
    if pd.api.types.is_numeric_dtype(col):
        if col.isna().any():
            raise ValueError(f"{target_col} contiene nulos antes de codificar.")
        values = set(col.dropna().unique().tolist())
        if values.issubset({0, 1}):
            return out
        raise ValueError(
            f"{target_col} numérica debe ser binaria {{0, 1}}; muestra de valores: {sorted(values)[:10]}"
        )
    invalid = col.notna() & ~col.isin(_EXPECTED_LABELS)
    if invalid.any():
        bad = col[invalid].unique()[:5].tolist()
        raise ValueError(f"Etiquetas de objetivo no permitidas en {target_col}: {bad}")
    out[target_col] = col.map(_TARGET_MAP)
    if out[target_col].isna().any():
        raise ValueError(f"Quedaron nulos tras mapear {target_col}; revisar tipos o etiquetas.")
    return out


def one_hot_encode(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN_DEFAULT, dummy_dtype=np.int8
) -> pd.DataFrame:
    """One-hot a columnas `object`, excluyendo la objetivo. Dummies compactas (`int8`)."""
    out = df.copy()
    categorical_cols = [
        c for c in out.select_dtypes(include="object").columns if c != target_col
    ]
    if not categorical_cols:
        return out
    return pd.get_dummies(
        out, columns=categorical_cols, drop_first=True, dtype=dummy_dtype
    )


def scale_numeric_columns(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN_DEFAULT
) -> tuple[pd.DataFrame, StandardScaler | None]:
    """
    Escala con StandardScaler todas las columnas numéricas excepto la objetivo.

    Devuelve el frame transformado y el scaler ajustado (o None si no hubo columnas que escalar).
    """
    out = df.copy()
    numeric_cols = [
        c
        for c in out.select_dtypes(include=np.number).columns
        if c != target_col
    ]
    if not numeric_cols:
        return out, None
    scaler = StandardScaler()
    out[numeric_cols] = scaler.fit_transform(out[numeric_cols].astype(np.float64))
    return out, scaler
