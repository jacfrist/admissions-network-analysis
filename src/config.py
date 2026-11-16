"""Shared file paths, so every script and notebook reads data from the same place.

Usage (from a notebook):  from src.config import HOSP_DIR
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw MIMIC-IV demo data (read-only; never edit these files)
MIMIC_DIR = ROOT / "mimic-iv-data"
HOSP_DIR = MIMIC_DIR / "hosp"

# Outputs
DATA_DIR = ROOT / "data"          # cleaned/intermediate tables
FIGURES_DIR = ROOT / "figures"    # plots for the report and slides
RESULTS_DIR = ROOT / "results"    # summary tables (descriptive stats, degree tables)

for d in (DATA_DIR, FIGURES_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)
