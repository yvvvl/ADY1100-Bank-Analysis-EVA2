"""Limpieza: valores desconocidos, pdays, outliers IQR y columnas con fuga de información."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.validation import UNKNOWN_TOKEN

PDAYS_NOT_CONTACTED_CODE = 999
IQR_MULTIPLIER = 1.5
QUANTILE_LOWER = 0.25
QUANTILE_UPPER = 0.75
MIN_SAMPLES_FOR_IQR = 4

COLUMN_PDAYS = "pdays"
COLUMN_PDAYS_CLEAN = "pdays_clean"
COLUMN_WAS_CONTACTED = "was_previously_contacted"
LEAKAGE_COLUMNS_DROP: tuple[str, ...] = ("duration", "pdays")


@dataclass(frozen=True)
class IqrBounds:
    q1: float
    q3: float
    iqr: float
    lower: float
    upper: float


def _iqr_bounds(series: pd.Series) -> IqrBounds | None:
    """Límites IQR para una serie numérica o None si no aplica."""
    clean = series.dropna()
    if clean.shape[0] < MIN_SAMPLES_FOR_IQR:
        return None
    q1 = float(series.quantile(QUANTILE_LOWER))
    q3 = float(series.quantile(QUANTILE_UPPER))
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    return IqrBounds(q1=q1, q3=q3, iqr=iqr, lower=lower, upper=upper)


def replace_unknown_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Sustituye `UNKNOWN_TOKEN` por NaN solo en columnas donde aparece."""
    out = df.copy()
    counts = out.astype(object).eq(UNKNOWN_TOKEN).sum()
    affected = counts[counts > 0].index.tolist()
    if affected:
        out[affected] = out[affected].replace(UNKNOWN_TOKEN, np.nan)
    return out


def impute_categorical_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa columnas categóricas (`object`) con la moda."""
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        if not out[col].isna().any():
            continue
        mode_series = out[col].mode(dropna=True)
        if mode_series.empty:
            continue
        out[col] = out[col].fillna(mode_series.iloc[0])
    return out


def impute_pdays_clean_median(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa `pdays_clean` con la mediana cuando hay NaN tras codificar no-contactado."""
    out = df.copy()
    if COLUMN_PDAYS_CLEAN not in out.columns:
        return out
    series = out[COLUMN_PDAYS_CLEAN]
    if not series.isna().any():
        return out
    median_val = series.median()
    out[COLUMN_PDAYS_CLEAN] = series.fillna(median_val)
    return out


def transform_pdays(df: pd.DataFrame) -> pd.DataFrame:
    """Separa el código especial de no-contactado de los días reales entre campañas."""
    out = df.copy()
    if COLUMN_PDAYS not in out.columns:
        return out
    pdays = out[COLUMN_PDAYS]
    out[COLUMN_WAS_CONTACTED] = np.where(pdays == PDAYS_NOT_CONTACTED_CODE, 0, 1)
    out[COLUMN_PDAYS_CLEAN] = pdays.replace(PDAYS_NOT_CONTACTED_CODE, np.nan)
    return out


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas definidas como fugas (`duration`, `pdays` original)."""
    out = df.copy()
    to_drop = [c for c in LEAKAGE_COLUMNS_DROP if c in out.columns]
    return out.drop(columns=to_drop) if to_drop else out


def detect_outliers_iqr(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Tabla de outliers por columna según regla IQR (solo columnas numéricas existentes)."""
    rows: list[dict[str, float | int | str]] = []
    n_rows = len(df)
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        bounds = _iqr_bounds(series)
        if bounds is None:
            continue
        outliers = int(((series < bounds.lower) | (series > bounds.upper)).sum())
        rows.append(
            {
                "column": col,
                "q1": bounds.q1,
                "q3": bounds.q3,
                "iqr": bounds.iqr,
                "lower_bound": bounds.lower,
                "upper_bound": bounds.upper,
                "outlier_count": outliers,
                "outlier_percent": round((outliers / n_rows) * 100, 2) if n_rows else 0.0,
            }
        )
    return pd.DataFrame(rows)


def cap_outliers_iqr(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Recorta valores a [lower, upper] por IQR en columnas elegibles."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        series = out[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        bounds = _iqr_bounds(series)
        if bounds is None:
            continue
        out[col] = series.clip(lower=bounds.lower, upper=bounds.upper)
    return out
