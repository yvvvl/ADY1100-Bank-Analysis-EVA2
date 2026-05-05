# ADY1100-EVA2-Bank-Marketing

Proyecto de análisis de marketing bancario para la evaluación EVA2 de ADY1100.

## Estructura del Proyecto

```
ADY1100-EVA2-Bank-Marketing/
├── data/raw/              ← Dataset original
├── data/processed/        ← Dataset limpio (bank_clean.csv)
├── notebooks/
│   └── informe_tecnico.ipynb   ← Notebook principal de análisis
├── src/
│   ├── load_data.py       ← Carga y validación inicial
│   ├── cleaning.py        ← Limpieza y tratamiento de outliers
│   ├── transform.py       ← Ingeniería de variables y escalado
│   ├── visualization.py   ← Funciones de visualización
│   └── validation.py      ← Checksum e integridad
├── outputs/figures/       ← Gráficos generados (PNG)
├── outputs/tables/        ← Reportes y tablas (CSV)
├── docs/
│   └── decisiones_preprocesamiento.md   ← Documentación de decisiones
├── README.md              ← Este archivo
└── requirements.txt       ← Dependencias del proyecto
```

## Instalación

1. Clonar el repositorio.
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Los módulos en `src/` están diseñados para ser importados desde el notebook `notebooks/informe_tecnico.ipynb`.
