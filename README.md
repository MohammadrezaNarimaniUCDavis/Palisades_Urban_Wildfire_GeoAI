# Palisades Urban Wildfire GeoAI

Public replication package for a **spatially validated GeoAI** analysis of
structure destruction in the **January 2025 Palisades Fire** (Los Angeles, California),
accompanying:

> *The spatial anatomy of urban wildfire vulnerability: a spatially validated GeoAI
> framework reveals the roles of building density and vegetation moisture in structure
> loss during the 2025 Palisades Fire*

**Authors:** Parastoo Farajpoor and Mohammadreza Narimani  
**Affiliation:** Department of Biological and Agricultural Engineering, UC Davis  
**Contact:** mnarimani@ucdavis.edu

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-zenodo-blue.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> **Zenodo DOI:** *(updated after deposit publication)*

This repository is the **public replication package**. It contains only audited `.py`
scripts and analysis-ready products needed to reproduce the paper’s core results.
The authors’ full working archive remains private.

---

## What this study does

1. Link **12,081** CAL FIRE DINS inspections (residential *n* = **9,883**) to a strictly
   **pre-fire** open predictor stack.
2. Build multi-ring (0–30 / 30–100 / 100–300 m) predictors from Sentinel-2 NDVI/NDMI,
   Landsat LST, LANDFIRE fuels, terrain, and a dated OSM building/road snapshot.
3. Compare nested **logistic** and **XGBoost** models under **random** vs **1 km
   spatial-block** cross-validation, with calibration and SHAP attribution.
4. Keep burn severity, recovery, and community/SVI analyses in a **separate impact track**
   (no post-fire leakage into predictors).

Claims are framed as **association / screening / spatially honest skill**, not causal
effects or official parcel risk scores.

### Locked headline numbers

| Metric | Value |
|---|---|
| Residential destroyed | 5,566 / 9,883 (56.3%) |
| Random-CV XGBoost ROC-AUC | **0.921 ± 0.007** |
| Spatial-CV XGBoost ROC-AUC | **0.753 ± 0.060** |
| Spatial-CV logistic ROC-AUC | **0.756 ± 0.092** |
| Optimism gap | ~0.17 AUC |
| Buildings within 100 m (OR / SD) | **4.12** (3.60–4.72) |
| NDMI 100–300 m (OR / SD) | **0.52** (0.47–0.57) |
| NDVI 30–100 m (OR / SD) | **1.74** (1.49–2.03) |

---

## Repository layout

```
Palisades_Urban_Wildfire_GeoAI/
├── config/                  # project + figure style
├── data/
│   ├── boundary/            # study area (shipped)
│   ├── derived/             # analysis-ready tables (shipped)
│   └── metadata/
├── src/                     # all pipeline code (.py)
│   ├── data/                # 01–06 acquisition
│   ├── preprocessing/       # 10–12
│   ├── features/            # 20
│   ├── analysis/            # 30–34
│   ├── visualization/       # 40–41
│   ├── manuscript/          # 50–51
│   └── utils/
├── results/
│   ├── tables/
│   ├── model_diagnostics/
│   └── figures/
├── docs/
├── tests/
├── run_pipeline.py
├── requirements.txt
├── requirements-pipeline.txt
├── CITATION.cff
└── LICENSE
```

---

## Quick start

```bash
git clone https://github.com/MohammadrezaNarimaniUCDavis/Palisades_Urban_Wildfire_GeoAI.git
cd Palisades_Urban_Wildfire_GeoAI

conda create -n palisades-geoai python=3.11 -y
conda activate palisades-geoai
pip install -r requirements.txt

python src/analysis/31_fit_models.py
python src/analysis/32_interpret.py
python src/manuscript/50_make_tables.py
python -m pytest tests -q
```

Full acquisition (GEE + Zenodo rasters): see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
and [`docs/DATA.md`](docs/DATA.md).

```bash
python run_pipeline.py
```

---

## Data availability

| Product | GitHub | Zenodo |
|---|---|---|
| Study-area boundary | yes | yes |
| Structure feature table (parquet/gpkg) | yes | yes |
| Manuscript tables + CV diagnostics | yes | yes |
| Final figures | yes | yes |
| OOF predictions + SHAP sample | yes | yes |
| Sentinel-2 / Landsat / LANDFIRE / DEM rasters | no | **yes** |
| Raw DINS / OSM snapshots | no | **yes** |

Modeled after the public replication style of
[Davis Urban Canopy GeoAI](https://github.com/MohammadrezaNarimaniUCDavis/Davis_Urban_Canopy_GeoAI).

---

## Citation

```bibtex
@dataset{Farajpoor_Narimani_Palisades_Urban_Wildfire_GeoAI_2026,
  author    = {Farajpoor, Parastoo and Narimani, Mohammadreza},
  title     = {Palisades Urban Wildfire GeoAI: structure-level predictors and spatially validated model outputs for the 2025 Palisades Fire},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}

@software{Farajpoor_Narimani_Palisades_Urban_Wildfire_GeoAI_code,
  author = {Farajpoor, Parastoo and Narimani, Mohammadreza},
  title  = {Palisades Urban Wildfire GeoAI: replication code},
  year   = {2026},
  url    = {https://github.com/MohammadrezaNarimaniUCDavis/Palisades_Urban_Wildfire_GeoAI},
  note   = {Companion dataset DOI: https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## License

Code: [MIT](LICENSE). Third-party geospatial inputs retain original licenses
([`docs/DATA.md`](docs/DATA.md), [`docs/NOTICE.md`](docs/NOTICE.md)).

---

## Contact

Mohammadreza Narimani — `mnarimani@ucdavis.edu` (University of California, Davis)
