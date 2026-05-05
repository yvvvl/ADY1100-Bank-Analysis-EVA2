"""Visualizaciones exploratorias (figuras aisladas, cierre explícito, I/O opcional)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save_fig(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "y",
    *,
    save_path: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> None:
    """Distribución de frecuencias de la variable objetivo."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        sns.countplot(data=df, x=target_col, ax=ax)
        ax.set_title("Distribución de la variable objetivo")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Frecuencia")
        fig.tight_layout()
        _save_fig(fig, save_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def plot_numeric_boxplots(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    *,
    save_prefix: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (7.0, 3.0),
) -> None:
    """Un boxplot por variable numérica elegible."""
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        fig, ax = plt.subplots(figsize=figsize)
        try:
            sns.boxplot(x=series, ax=ax)
            ax.set_title(f"Boxplot de {col}")
            fig.tight_layout()
            if save_prefix is not None:
                prefix = Path(save_prefix)
                out_path = prefix.parent / f"{prefix.name}_{col}.png"
                _save_fig(fig, out_path)
            if show:
                plt.show()
        finally:
            plt.close(fig)


def plot_pie_distribution(
    df: pd.DataFrame,
    col: str,
    *,
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (6.0, 6.0),
    autopct: str = "%1.1f%%",
) -> None:
    """Gráfico de torta para proporciones de una variable categórica."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        counts = df[col].value_counts()
        ax.pie(counts, labels=counts.index.astype(str), autopct=autopct, startangle=90)
        ax.set_title(title or f"Proporción de {col}")
        ax.axis("equal")
        fig.tight_layout()
        _save_fig(fig, save_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def plot_countplot(
    df: pd.DataFrame,
    col: str,
    *,
    hue: str | None = None,
    orient: str = "v",
    order: Sequence | None = None,
    palette: str | None = None,
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (10.0, 5.0),
) -> None:
    """Barras de conteo (vertical u horizontal); opcional agrupación con hue."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        if orient == "h":
            sns.countplot(data=df, y=col, hue=hue, order=order, palette=palette, ax=ax)
        else:
            sns.countplot(data=df, x=col, hue=hue, order=order, palette=palette, ax=ax)
        ax.set_title(title or f"Conteos de {col}")
        fig.tight_layout()
        _save_fig(fig, save_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def plot_barplot_rate(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "y",
    *,
    order: Sequence | None = None,
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (10.0, 5.0),
) -> None:
    """Tasa media del objetivo binario (yes/no -> 1/0) por categoría de group_col."""
    work = df.copy()
    if work[target_col].dtype == object:
        work["_target_bin"] = work[target_col].map({"yes": 1, "no": 0})
        y_use = "_target_bin"
    else:
        y_use = target_col
    fig, ax = plt.subplots(figsize=figsize)
    try:
        sns.barplot(data=work, x=group_col, y=y_use, order=order, ax=ax, errorbar=("ci", 95))
        ax.set_ylabel("Tasa media de suscripción")
        ax.set_xlabel(group_col)
        ax.set_title(title or f"Tasa de suscripción por {group_col}")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        _save_fig(fig, save_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def plot_histogram_kde(
    df: pd.DataFrame,
    col: str,
    *,
    kde: bool = True,
    bins: int | str = "auto",
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> None:
    """Histograma opcionalmente con KDE para variables numéricas."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        sns.histplot(data=df, x=col, kde=kde, bins=bins, ax=ax)
        ax.set_title(title or f"Distribución de {col}")
        fig.tight_layout()
        _save_fig(fig, save_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    *,
    mask_upper_triangle: bool = True,
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (12.0, 10.0),
) -> None:
    """Heatmap de Pearson sobre columnas numéricas del DataFrame."""
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return
    corr = num.corr()
    mask = None
    if mask_upper_triangle:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=figsize)
    try:
        sns.heatmap(
            corr,
            mask=mask,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(title or "Correlaciones entre variables numéricas")
        fig.tight_layout()
        _save_fig(fig, save_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def plot_violin(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    title: str | None = None,
    palette: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> None:
    """Violinplot: distribución de y_col segmentada por x_col."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        sns.violinplot(data=df, x=x_col, y=y_col, palette=palette, ax=ax, inner="box")
        ax.set_title(title or f"{y_col} según {x_col}")
        fig.tight_layout()
        _save_fig(fig, save_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def plot_boxplot_bivariate(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    title: str | None = None,
    palette: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> None:
    """Boxplot de una variable numérica (y) por categorías en x."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        sns.boxplot(data=df, x=x_col, y=y_col, palette=palette, ax=ax)
        ax.set_title(title or f"{y_col} por {x_col}")
        fig.tight_layout()
        _save_fig(fig, save_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def plot_macro_trends(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    group_col: str = "month",
    *,
    month_order: Sequence[str] | None = None,
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (12.0, 6.0),
) -> None:
    """Promedio agregado de indicadores macro por valor de group_col (típicamente mes)."""
    default_months = (
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    )
    order = list(month_order) if month_order else list(default_months)
    agg = df.groupby(group_col, observed=False)[list(value_cols)].mean()
    agg = agg.reindex([m for m in order if m in agg.index])
    fig, ax = plt.subplots(figsize=figsize)
    try:
        for c in value_cols:
            if c in agg.columns:
                ax.plot(agg.index.astype(str), agg[c].values, marker="o", label=c)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel(group_col)
        ax.set_ylabel("Valor medio agregado")
        ax.set_title(title or "Indicadores macroeconómicos por mes (promedio)")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        _save_fig(fig, save_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def plot_scatter_with_hue(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    *,
    alpha: float = 0.35,
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (9.0, 6.0),
) -> None:
    """Dispersión bivariada con color por una tercera variable categórica."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=alpha, ax=ax)
        ax.set_title(title or f"{y_col} vs {x_col} por {hue_col}")
        fig.tight_layout()
        _save_fig(fig, save_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)
