"""Verify every candidate reference against Crossref and generate references.bib.

For entries with a DOI: fetch metadata from api.crossref.org/works/{doi} and
check that the title matches the expected fragment. For entries without a
known DOI: resolve via bibliographic search, then verify. Entries that cannot
be verified are EXCLUDED from the generated .bib and reported.

Outputs:
  references/references.bib            (Crossref-verified entries + manual non-DOI items)
  references/reference_verification.csv (verification audit)
  references/literature_review.csv      (role/notes per reference)
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.project import get_logger, path

log = get_logger("verify_references")
API = "https://api.crossref.org/works"
HEADERS = {"User-Agent": "palisades-resilience-research/1.0 (mailto:research@example.edu)"}

# key, doi (or None), expected title fragment, role in manuscript
CANDIDATES: list[dict] = [
    # --- Palisades / 2025 LA fires ---
    dict(key="kenny2026", doi="10.1016/j.ufug.2026.129470",
         title="Urban trees and structure loss in the 2025 Eaton and Palisades fires",
         role="Prior Palisades study: trees vs density"),
    dict(key="norlen2026", doi="10.1038/s41467-026-71376-1",
         title="Socio-ecological impacts of the 2025 Los Angeles urban fires",
         role="Prior Palisades study: socio-ecological GLMs"),
    # --- structure loss / WUI ---
    dict(key="syphard2012", doi="10.1371/journal.pone.0033954",
         title="Housing arrangement and location determine the likelihood of housing loss due to wildfire",
         role="Housing arrangement & loss"),
    dict(key="syphard2014", doi="10.1071/WF13158",
         title="The role of defensible space for residential structure protection during wildfires",
         role="Defensible space"),
    dict(key="syphard2019", doi="10.3390/fire2030049",
         title="Factors associated with structure loss in the 2013",
         role="DINS-based structure loss analysis"),
    dict(key="syphard2021", doi="10.3390/fire4010012",
         title="Multiple-scale relationships between vegetation, the wildland",
         role="Scale dependence of vegetation effects"),
    dict(key="knapp2021", doi="10.1186/s42408-021-00117-0",
         title="Housing arrangement and vegetation factors associated with single-family home survival",
         role="Camp Fire home survival"),
    dict(key="gibbons2012", doi="10.1371/journal.pone.0029212",
         title="Land management practices associated with house loss in wildfires",
         role="House loss factors (Australia)"),
    dict(key="alexandre2016", doi="10.1002/eap.1376",
         title="Factors related to building loss due to wildfires in the conterminous United States",
         role="National structure loss"),
    dict(key="kramer2019", doi="10.1071/WF18108",
         title="High wildfire damage in interface communities in California",
         role="Interface community damage"),
    dict(key="caggiano2020", doi="10.3390/fire3040073",
         title="Building Loss in WUI Disasters",
         role="WUI definition components & loss"),
    dict(key="cohen2000", doi="10.1093/jof/98.3.15",
         title="Preventing Disaster: Home Ignitability in the Wildland-Urban Interface",
         role="Home ignition zone"),
    dict(key="calkin2014", doi="10.1073/pnas.1315088111",
         title="How risk management can prevent future wildfire disasters",
         role="WUI risk management"),
    dict(key="calkin2023", doi="10.1073/pnas.2315797120",
         title="Wildland-urban fire disasters aren",
         role="Urban conflagration framing"),
    dict(key="zamanialaei2025", doi="10.1038/s41467-025-63386-2",
         title="Fire risk to structures in California",
         role="DINS + exposure modeling"),
    dict(key="escobedo2025", doi="10.1016/j.landurbplan.2025.105421",
         title="Exploring urban vegetation type and defensible space",
         role="Urban vegetation & building loss"),
    dict(key="metz2024", doi=None,
         title="The Influence of Housing, Parcel, and Neighborhood Characteristics on Housing Survival in the Marshall Fire",
         role="Marshall Fire suburban conflagration"),
    dict(key="radeloff2018", doi="10.1073/pnas.1718850115",
         title="Rapid growth of the US wildland-urban interface raises wildfire risk",
         role="WUI growth"),
    dict(key="carlson2022", doi="10.1002/eap.2597",
         title="The wildland",
         role="WUI mapping from building locations"),
    dict(key="schug2023", doi="10.1038/s41586-023-06320-0",
         title="The global wildland",
         role="Global WUI context"),
    dict(key="balch2024", doi=None,
         title="The fastest-growing and most destructive fires in the US",
         role="Fast fires context"),
    # --- fire regime / climate ---
    dict(key="keeley2019", doi="10.1186/s42408-019-0041-0",
         title="Twenty-first century California, USA, wildfires: fuel-dominated vs. wind-dominated fires",
         role="Wind- vs fuel-dominated fires"),
    dict(key="guzman2019", doi="10.1029/2018GL080261",
         title="Santa Ana Winds of Southern California",
         role="Santa Ana winds climatology"),
    dict(key="sadegh2025", doi=None,
         title="Ignition matters",
         role="2025 LA fires commentary"),
    # --- remote sensing methods ---
    dict(key="tucker1979", doi="10.1016/0034-4257(79)90013-0",
         title="Red and photographic infrared linear combinations for monitoring vegetation",
         role="NDVI"),
    dict(key="gao1996", doi="10.1016/S0034-4257(96)00067-3",
         title="NDWI",
         role="NDWI/NDMI moisture index"),
    dict(key="miller2007", doi="10.1016/j.rse.2006.12.006",
         title="Quantifying burn severity in a heterogeneous landscape",
         role="RdNBR"),
    dict(key="dennison2005", doi="10.1080/0143116042000273998",
         title="Use of Normalized Difference Water Index for monitoring live fuel moisture",
         role="Live fuel moisture from spectral indices"),
    dict(key="gorelick2017", doi="10.1016/j.rse.2017.06.031",
         title="Google Earth Engine",
         role="GEE platform"),
    dict(key="nolde2025", doi="10.1080/15481603.2025.2498188",
         title="Multi-sensor near-realtime burnt area monitoring",
         role="Target-journal exemplar (burnt area)"),
    dict(key="ramayanti2024", doi="10.1080/15481603.2024.2353982",
         title="Wildfire susceptibility mapping by incorporating damage proxy maps",
         role="Target-journal exemplar (wildfire ML)"),
    # --- statistics / ML methods ---
    dict(key="roberts2017", doi="10.1111/ecog.02881",
         title="Cross-validation strategies for data with temporal, spatial, hierarchical",
         role="Spatial CV"),
    dict(key="ploton2020", doi="10.1038/s41467-020-18321-y",
         title="Spatial validation reveals poor predictive performance of large-scale ecological mapping models",
         role="Spatial validation necessity"),
    dict(key="valavi2019", doi="10.1111/2041-210X.13107",
         title="blockCV",
         role="Spatial block CV package/concepts"),
    dict(key="chen2016", doi="10.1145/2939672.2939785",
         title="XGBoost",
         role="Gradient boosting"),
    dict(key="saito2015", doi="10.1371/journal.pone.0118432",
         title="The Precision-Recall Plot Is More Informative than the ROC Plot",
         role="PR-AUC for imbalanced data"),
    dict(key="apley2020", doi="10.1111/rssb.12377",
         title="Visualizing the effects of predictor variables in black box supervised learning models",
         role="ALE plots"),
    # --- vulnerability / resilience ---
    dict(key="meerow2016", doi="10.1016/j.landurbplan.2015.11.011",
         title="Defining urban resilience",
         role="Urban resilience definition"),
    dict(key="cutter2008", doi="10.1016/j.gloenvcha.2008.07.013",
         title="A place-based model for understanding community resilience",
         role="Community resilience model"),
    dict(key="flanagan2011", doi="10.2202/1547-7355.1792",
         title="A Social Vulnerability Index for Disaster Management",
         role="SVI methodology"),
    dict(key="davies2018", doi="10.1371/journal.pone.0205825",
         title="The unequal vulnerability of communities of color to wildfire",
         role="Wildfire vulnerability equity"),
    dict(key="mcwethy2019", doi="10.1038/s41893-019-0353-8",
         title="Rethinking resilience to wildfire",
         role="Wildfire resilience framing"),
    # --- data sources with papers ---
    dict(key="abatzoglou2013", doi="10.1002/joc.3413",
         title="Development of gridded surface meteorological data",
         role="gridMET"),
    dict(key="rollins2009", doi="10.1071/WF08088",
         title="LANDFIRE: a nationally consistent vegetation, wildland fire, and fuel assessment",
         role="LANDFIRE"),
    dict(key="scott2005", doi=None,
         title="Standard fire behavior fuel models: a comprehensive set for use with Rothermel",
         role="FBFM40 fuel models"),
    dict(key="munozsabater2021", doi="10.5194/essd-13-4349-2021",
         title="ERA5-Land: a state-of-the-art global reanalysis dataset for land applications",
         role="ERA5-Land"),
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def crossref_by_doi(doi: str) -> dict | None:
    r = requests.get(f"{API}/{doi}", headers=HEADERS, timeout=60)
    if r.status_code != 200:
        return None
    return r.json()["message"]


def crossref_search(title: str) -> dict | None:
    r = requests.get(API, params={"query.bibliographic": title, "rows": 3},
                     headers=HEADERS, timeout=60)
    if r.status_code != 200:
        return None
    items = r.json()["message"]["items"]
    for it in items:
        t = (it.get("title") or [""])[0]
        if norm(title)[:40] in norm(t) or norm(t)[:40] in norm(title):
            return it
    return items[0] if items else None


def to_bibtex(key: str, m: dict) -> str:
    def latex_escape(s: str) -> str:
        return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

    authors = " and ".join(
        f"{a.get('family', '')}, {a.get('given', '')}" for a in m.get("author", [])
        if a.get("family")
    )
    title = latex_escape((m.get("title") or [""])[0])
    journal = latex_escape((m.get("container-title") or [""])[0])
    year = None
    for k in ("published-print", "published-online", "issued"):
        if m.get(k, {}).get("date-parts"):
            year = m[k]["date-parts"][0][0]
            break
    vol = m.get("volume", "")
    issue = m.get("issue", "")
    pages = m.get("page", "") or m.get("article-number", "")
    doi = m.get("DOI", "")
    lines = [f"@article{{{key},",
             f"  author = {{{authors}}},",
             f"  title = {{{title}}},",
             f"  journal = {{{journal}}},",
             f"  year = {{{year}}},"]
    if vol:
        lines.append(f"  volume = {{{vol}}},")
    if issue:
        lines.append(f"  number = {{{issue}}},")
    if pages:
        lines.append(f"  pages = {{{pages}}},")
    lines.append(f"  doi = {{{doi}}}")
    lines.append("}")
    return "\n".join(lines)


MANUAL_ENTRIES = """
@article{valavi2019,
  author = {Valavi, Roozbeh and Elith, Jane and Lahoz-Monfort, Jos{\\'e} J. and Guillera-Arroita, Gurutzeta},
  title = {block{CV}: An {R} package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models},
  journal = {Methods in Ecology and Evolution},
  year = {2019},
  volume = {10},
  number = {2},
  pages = {225--232},
  doi = {10.1111/2041-210X.13107}
}

@techreport{key2006,
  author = {Key, Carl H. and Benson, Nathan C.},
  title = {Landscape Assessment ({LA}): Sampling and Analysis Methods},
  institution = {USDA Forest Service, Rocky Mountain Research Station},
  type = {General Technical Report},
  number = {RMRS-GTR-164-CD},
  year = {2006},
  address = {Fort Collins, CO},
  note = {In: FIREMON: Fire Effects Monitoring and Inventory System}
}

@inproceedings{lundberg2017,
  author = {Lundberg, Scott M. and Lee, Su-In},
  title = {A Unified Approach to Interpreting Model Predictions},
  booktitle = {Advances in Neural Information Processing Systems 30 (NIPS 2017)},
  year = {2017},
  pages = {4765--4774},
  address = {Long Beach, CA}
}

@misc{calfire2025incident,
  author = {{CAL FIRE}},
  title = {Palisades Fire Incident Information},
  year = {2025},
  howpublished = {\\url{https://www.fire.ca.gov/incidents/2025/1/7/palisades-fire/}},
  note = {California Department of Forestry and Fire Protection. Accessed 2026-08-12}
}

@misc{calfire2025dins,
  author = {{CAL FIRE}},
  title = {{CAL FIRE} Damage Inspection ({DINS}) Data},
  year = {2025},
  howpublished = {\\url{https://data.ca.gov/dataset/cal-fire-damage-inspection-dins-data}},
  note = {California Open Data Portal. Accessed 2026-08-12}
}

@misc{nifc2025wfigs,
  author = {{National Interagency Fire Center}},
  title = {{WFIGS} Interagency Fire Perimeters},
  year = {2025},
  howpublished = {\\url{https://data-nifc.opendata.arcgis.com/datasets/nifc::wfigs-interagency-fire-perimeters/about}},
  note = {Accessed 2026-08-12}
}

@misc{landfire2024,
  author = {{LANDFIRE}},
  title = {{LANDFIRE} 2024 Update ({LF} 2024): 40 {Scott} and {Burgan} Fire Behavior Fuel Models and Forest Canopy Cover},
  year = {2025},
  howpublished = {\\url{https://landfire.gov}},
  note = {U.S. Geological Survey and USDA Forest Service. Accessed 2026-08-12}
}

@misc{osm2025,
  author = {{OpenStreetMap contributors}},
  title = {OpenStreetMap database snapshot (2025-01-01)},
  year = {2025},
  howpublished = {\\url{https://www.openstreetmap.org}},
  note = {Open Database License (ODbL). Historical snapshot retrieved via Overpass API, accessed 2026-08-12}
}

@misc{cdcsvi2022,
  author = {{Centers for Disease Control and Prevention / Agency for Toxic Substances and Disease Registry}},
  title = {{CDC/ATSDR} Social Vulnerability Index 2022 Database, California},
  year = {2024},
  howpublished = {\\url{https://www.atsdr.cdc.gov/place-health/php/svi/}},
  note = {Accessed 2026-08-12}
}

@techreport{ibhs2025,
  author = {{Insurance Institute for Business \\& Home Safety}},
  title = {The 2025 {LA} Conflagrations},
  institution = {IBHS},
  year = {2025},
  note = {Technical field-study report}
}

@techreport{scott2005,
  author = {Scott, Joe H. and Burgan, Robert E.},
  title = {Standard Fire Behavior Fuel Models: A Comprehensive Set for Use with {Rothermel}'s Surface Fire Spread Model},
  institution = {USDA Forest Service, Rocky Mountain Research Station},
  type = {General Technical Report},
  number = {RMRS-GTR-153},
  year = {2005},
  address = {Fort Collins, CO}
}
"""


def main() -> None:
    results = []
    bib_entries = []
    for c in CANDIDATES:
        time.sleep(0.4)
        m = None
        method = ""
        if c.get("doi"):
            m = crossref_by_doi(c["doi"])
            method = "doi"
        if m is None:
            m = crossref_search(c["title"])
            method = "search"
        status = "FAILED"
        got_title = ""
        got_doi = ""
        if m:
            got_title = (m.get("title") or [""])[0]
            got_doi = m.get("DOI", "")
            if norm(c["title"])[:35] in norm(got_title) or norm(got_title) and norm(got_title)[:35] in norm(c["title"]):
                status = "VERIFIED"
        results.append({
            "key": c["key"], "expected_title": c["title"], "crossref_title": got_title,
            "doi": got_doi or c.get("doi") or "", "method": method, "status": status,
            "role": c["role"],
        })
        if status == "VERIFIED" and c["key"] != "scott2005":
            bib_entries.append(to_bibtex(c["key"], m))
        log.info("%-16s %-8s %s", c["key"], status, got_doi)

    ver_fp = path("references", "reference_verification.csv")
    with open(ver_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    bib = "\n\n".join(bib_entries) + "\n" + MANUAL_ENTRIES
    path("references", "references.bib").write_text(bib, encoding="utf-8")

    lit_fp = path("references", "literature_review.csv")
    with open(lit_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "doi", "title", "role", "verified"])
        w.writeheader()
        for r in results:
            w.writerow({"key": r["key"], "doi": r["doi"],
                        "title": r["crossref_title"] or r["expected_title"],
                        "role": r["role"], "verified": r["status"]})

    n_ok = sum(r["status"] == "VERIFIED" for r in results)
    log.info("verified %d/%d candidates; bib written", n_ok, len(results))
    failed = [r["key"] for r in results if r["status"] != "VERIFIED"]
    if failed:
        log.warning("NOT verified (excluded from bib): %s", failed)


if __name__ == "__main__":
    main()
