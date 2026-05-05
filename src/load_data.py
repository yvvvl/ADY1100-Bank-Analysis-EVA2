"""
load_data.py — Módulo de carga y validación inicial del dataset.
Responsable: Lucas Moncada

Carga el archivo bank-additional-full.csv y realiza una validación
estructural básica (columnas esperadas, tipos, shape).
"""

import pandas as pd
from pathlib import Path


# Columnas esperadas según el diccionario de datos del caso de negocio
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "y"
]


def cargar_dataset(ruta: Path) -> pd.DataFrame:
    """
    Carga el dataset bank-additional-full.csv desde la ruta especificada.

    El archivo usa punto y coma (;) como separador, por lo que se debe
    especificar sep=';'. Se preservan los tipos de datos originales para
    que el módulo de limpieza los procese posteriormente.

    Args:
        ruta: Ruta al archivo CSV (ej: Path("data/raw/bank-additional-full.csv")).

    Returns:
        DataFrame con los 41,188 registros y 21 columnas cargados sin modificar.

    Example:
        >>> df = cargar_dataset(Path("data/raw/bank-additional-full.csv"))
        >>> print(df.shape)  # (41188, 21)
    """
    try:
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

        df = pd.read_csv(ruta, sep=";")
        print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        return df

    except FileNotFoundError as e:
        print(f"[cargar_dataset] Error de archivo: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[cargar_dataset] Error inesperado: {e}")
        return pd.DataFrame()


def validar_estructura(df: pd.DataFrame) -> bool:
    """
    Valida que el DataFrame tenga las columnas y el shape esperados.

    Compara las columnas presentes contra EXPECTED_COLUMNS y reporta
    cualquier discrepancia. No modifica el DataFrame.

    Args:
        df: DataFrame cargado desde el CSV original.

    Returns:
        True si la estructura es correcta, False si hay diferencias.

    Example:
        >>> es_valido = validar_estructura(df)
        >>> # ✅ Estructura válida: 21 columnas, 41,188 filas
    """
    try:
        columnas_presentes = set(df.columns)
        columnas_esperadas = set(EXPECTED_COLUMNS)

        faltantes = columnas_esperadas - columnas_presentes
        extra = columnas_presentes - columnas_esperadas

        if faltantes:
            print(f"⚠️  Columnas faltantes: {faltantes}")
        if extra:
            print(f"⚠️  Columnas extra no esperadas: {extra}")

        if not faltantes and not extra:
            print(f"✅ Estructura válida: {df.shape[1]} columnas, {df.shape[0]:,} filas")
            return True

        return False

    except Exception as e:
        print(f"[validar_estructura] Error: {e}")
        return False


def resumen_inicial(df: pd.DataFrame) -> None:
    """
    Imprime un resumen descriptivo del dataset recién cargado.

    Muestra tipos de datos, conteo de nulos y estadísticas básicas
    para dar un primer vistazo antes de la limpieza profunda.

    Args:
        df: DataFrame con los datos crudos.

    Example:
        >>> resumen_inicial(df_raw)
    """
    try:
        print("\n" + "="*60)
        print("RESUMEN INICIAL DEL DATASET")
        print("="*60)
        print(f"\nShape: {df.shape}")
        print(f"\n{'Columna':<20} {'Tipo':<15} {'Nulos':<10} {'Únicos'}")
        print("-"*60)
        for col in df.columns:
            nulos = df[col].isna().sum()
            unicos = df[col].nunique()
            print(f"{col:<20} {str(df[col].dtype):<15} {nulos:<10} {unicos}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"[resumen_inicial] Error: {e}")