from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Root directory of the MARS-Summer-Research repository
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

FEI_DIR = RAW_DATA_DIR / "FEI"

# ============================================================
# PROJECT DIRECTORIES
# ============================================================

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
POSTER_DIR = PROJECT_ROOT / "poster"
PRESENTATIONS_DIR = PROJECT_ROOT / "presentations"
LITERATURE_DIR = PROJECT_ROOT / "literature"


# ============================================================
# CREATE GENERATED DIRECTORIES IF NEEDED
# ============================================================

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)