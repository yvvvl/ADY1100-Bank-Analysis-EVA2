# Reporte de limpieza — Bank Marketing

## Dataset
- **Nombre:** bank-additional-full.csv
- **Ruta original (relativa):** `data/raw/bank-additional-full.csv`
- **Checksum MD5 (raw):** `f6cb2c1256ffe2836b36df321f46e92c`

## Dimensiones
| Etapa | Filas | Columnas |
|-------|-------|----------|
| Inicial | 41188 | 21 |
| Final | 41188 | 48 |

## Columnas con `unknown` (inicial)
```
           unknown_count  unknown_percent
job                  330         0.801204
marital               80         0.194231
education           1731         4.202680
default             8597        20.872584
housing              990         2.403613
loan                 990         2.403613
```

## Estrategias
1. **Valores faltantes (`unknown`):** sustitución por NaN solo donde aparece la cadena `unknown`; imputación de categóricas con **moda**.
2. **`pdays`:** valor **999** tratado como "no contactado"; variables derivadas `was_previously_contacted` y `pdays_clean`; imputación de `pdays_clean` con **mediana**; eliminación de `pdays` original.
3. **`duration`:** eliminada por **fuga de información** respecto del resultado de la llamada.
4. **Outliers:** detección IQR en variables numéricas; **capping** aplicado solo a `age`, `campaign`, `previous`, `pdays_clean`. Indicadores macro **no capeados** (contexto económico real); solo escalados al final.
5. **Codificación:** `y` como 0/1; demás categóricas con **one-hot** (`drop_first=True`).
6. **Escalamiento:** **StandardScaler** en todas las columnas numéricas excepto `y`.

## Variables escaladas (todas las numéricas excepto `y`)
Total columnas predictoras numéricas (excluye objetivo `y`): 47.

## Archivos generados
- `data/processed/bank_cleaned.csv`
- `outputs/limpieza_resumen.csv`
- Figuras en `outputs/figures/`