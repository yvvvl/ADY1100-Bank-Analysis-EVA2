# Decisiones de preprocesamiento

Documento de respaldo para el proyecto **Bank Marketing** (ADY1100 · EVA2). Las decisiones detalladas se aplican en [`notebooks/01_limpieza_bank_marketing.ipynb`](../notebooks/01_limpieza_bank_marketing.ipynb) y en los módulos de [`src/`](../src/).

## 1. Tratamiento de valores faltantes nativos (`NaN`)

- El CSV original no presenta celdas vacías estándar en cantidad relevante; el foco del tratamiento está en valores codificados como texto ausente (ver sección 2).
- Tras todas las transformaciones, el conjunto procesado (`data/processed/bank_cleaned.csv`) se valida sin nulos antes del guardado.

## 2. Gestión del valor `unknown`

- La cadena **`unknown`** se interpreta como dato faltante en atributos categóricos (por ejemplo `job`, `marital`, `education`, `default`, `housing`, `loan`).
- **Estrategia:** reemplazo por `NaN` solo en columnas donde aparece `unknown`; imputación de categóricas con la **moda** por columna, preservando el número de filas (sin eliminación masiva de registros).

## 3. Outliers y variables extremas

- **Detección:** regla **IQR** (1.5 × rango intercuartílico) para cuantificar valores extremos en variables numéricas.
- **Capping (recorte):** aplicado solo a `age`, `campaign`, `previous` y `pdays_clean`, donde los extremos pueden deberse a errores operativos o pocos contactos atípicos.
- **Sin capping:** indicadores macroeconómicos (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`), porque representan condiciones reales del periodo 2008–2010.
- **`pdays = 999`:** no se trata como outlier numérico; significa “no contactado en campaña anterior” y se transforma en variables derivadas antes de cualquier recorte.

## 4. Transformación de variables

- **`pdays`:** se crean `was_previously_contacted` (0/1) y `pdays_clean` (999 → `NaN`); `pdays_clean` se imputa con la **mediana** de días entre contactos válidos.
- **Fuga de información:** se eliminan **`duration`** y la columna original **`pdays`** del conjunto final (`duration` se conoce tras la llamada; sesga conclusiones predictivas realistas).
- **Objetivo `y`:** codificación binaria `yes` → 1, `no` → 0.
- **Categóricas:** *one-hot encoding* con `pd.get_dummies(..., drop_first=True)` y tipo entero compacto para las dummies.
- **Escalamiento:** `StandardScaler` sobre todas las columnas numéricas **excepto** `y`; el objeto scaler ajustado puede guardarse para aplicar la misma transformación a datos nuevos.

Para el plan detallado de la entrega de visualización (EVA2), ver [`plan_eva2.md`](plan_eva2.md).
