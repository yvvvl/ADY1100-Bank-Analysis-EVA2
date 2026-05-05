# ADY1100 · EVA2 — Bank Marketing

Proyecto de **preprocesamiento** y **informe técnico de visualizaciones** sobre el dataset *Bank Marketing* (campañas 2008–2010), alineado al caso **Banco Financiero Global**.

## Estructura del proyecto

```
ADY1100-Bank-Analysis-EVA2/
├── data/raw/                    ← bank-additional-full.csv (original)
├── data/processed/              ← bank_cleaned.csv (tras limpieza)
├── notebooks/
│   ├── 01_limpieza_bank_marketing.ipynb   ← Pipeline de limpieza y exportación
│   └── informe_tecnico.ipynb              ← Informe EVA2 (≥17 gráficos, referencias)
├── src/
│   ├── load_data.py             ← Rutas del proyecto y load_bank_data()
│   ├── validation.py          ← Columnas esperadas, resúmenes, unknown, MD5
│   ├── cleaning.py             ← unknown, pdays, IQR, eliminación duration/pdays
│   ├── transform.py            ← encode_target, one_hot_encode, scale_numeric_columns
│   └── visualization.py       ← Funciones de gráficos reutilizables
├── outputs/
│   ├── figures/                ← PNG del informe (g01…g17) y limpieza
│   └── limpieza_resumen.csv / .md
├── docs/
│   ├── plan_eva2.md            ← Plan de acción EVA2
│   └── decisiones_preprocesamiento.md
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

Se recomienda un entorno virtual (`.venv`).

## Flujo de trabajo sugerido

1. **Limpieza:** abrir y ejecutar [`notebooks/01_limpieza_bank_marketing.ipynb`](notebooks/01_limpieza_bank_marketing.ipynb) desde la carpeta `notebooks/` (o la raíz del proyecto). Genera `data/processed/bank_cleaned.csv` y reportes en `outputs/`.
2. **Informe EVA2:** ejecutar [`notebooks/informe_tecnico.ipynb`](notebooks/informe_tecnico.ipynb). Requiere que exista `bank_cleaned.csv`. Exporta figuras `g01`–`g17` en `outputs/figures/`.

## Uso programático (`src`)

Con la raíz del proyecto en `sys.path`:

```python
from src.load_data import load_bank_data, get_project_root
from src.visualization import plot_target_distribution, plot_correlation_heatmap
```

Desde la carpeta `notebooks/`, suele bastar con:

```python
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))
```

## Referencias del dataset

Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*, 62, 22–31.
