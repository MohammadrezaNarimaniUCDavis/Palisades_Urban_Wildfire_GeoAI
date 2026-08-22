# Reproducibility

## Environment

```bash
conda create -n palisades-geoai python=3.11 -y
conda activate palisades-geoai
pip install -r requirements.txt
```

Full acquisition extras:

```bash
pip install -r requirements-pipeline.txt
earthengine authenticate
```

## Refit from shipped derived table

```bash
python src/analysis/31_fit_models.py
python src/analysis/32_interpret.py
python src/manuscript/50_make_tables.py
python -m pytest tests -q
```

## Full pipeline

```bash
python run_pipeline.py
```

## Locked targets

| Setting | Model | ROC-AUC |
|---|---|---|
| Random CV | XGBoost M3 | 0.921 ± 0.007 |
| Spatial 1 km CV | XGBoost M3 | 0.753 ± 0.060 |
| Spatial 1 km CV | Logistic M3 | 0.756 ± 0.092 |
| Buildings @ 100 m | OR / SD | 4.12 (3.60–4.72) |
| NDMI 100–300 m | OR / SD | 0.52 (0.47–0.57) |
| NDVI 30–100 m | OR / SD | 1.74 (1.49–2.03) |
