# Data sources and licenses

## Inputs

| Dataset | Provider | Role | Terms |
|---|---|---|---|
| CAL FIRE DINS (Palisades) | CAL FIRE | Damage labels | CA open government data |
| WFIGS / NIFC perimeter | NIFC | Study extent | Public |
| Sentinel-2 L2A | Copernicus / ESA | NDVI, NDMI, burn indices | Copernicus open data |
| Landsat 8/9 ST | USGS | Pre-fire LST | Public domain |
| LANDFIRE 2024 | USDA / USDOI | Fuels / canopy | Public |
| USGS 3DEP DEM | USGS | Terrain | Public domain |
| OpenStreetMap (attic 2025-01-01) | OSM contributors | Buildings / roads | ODbL |
| CDC/ATSDR SVI 2022 | CDC | Community context | Public |
| Census TIGER | US Census | Tract geometry | Public |

## Locked analysis population

- Raw DINS: **12,137**
- Excluded Inaccessible: **56**
- Modeling universe: **12,081**
- Residential primary sample: **9,883** (5,566 destroyed; 56.3%)

## Zenodo companion

Large rasters and provenance snapshots are published on Zenodo. After download,
place rasters under `data/raw/` as described in the Zenodo record README.
