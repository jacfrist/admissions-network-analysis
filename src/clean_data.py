"""Clean the MIMIC-IV demo tables and save analysis-ready tables to data/.

Run from the repo root:   python3 -m src.clean_data

The raw files in mimic-iv-data/ are only read, never modified.

Outputs (in data/):
    admissions_clean.csv  one row per admission in the study cohorts
    unit_stays.csv        each admission's ordered sequence of care-unit stays
    transitions.csv       unit-to-unit moves (the edges of the networks)
"""
import pandas as pd

from src.config import DATA_DIR, HOSP_DIR

# ---------------------------------------------------------------------------
# Cohort definitions: which admission_type labels go in each group.
# Primary ("broad") definition. Anything not listed (the observation types)
# is excluded from the study.
# ---------------------------------------------------------------------------
EMERGENCY_TYPES = ["EW EMER.", "DIRECT EMER.", "URGENT"]
ELECTIVE_TYPES = ["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]

# Strict definition, kept as a flag for a sensitivity analysis.
STRICT_EMERGENCY_TYPES = ["EW EMER.", "DIRECT EMER."]
STRICT_ELECTIVE_TYPES = ["ELECTIVE"]

# Care units that are not real locations. Stays in these units are kept in
# unit_stays.csv, but any transition into or out of them is not used as an edge.
NON_UNITS = ["Unknown"]


def load_raw():
    admissions = pd.read_csv(
        HOSP_DIR / "admissions.csv.gz", parse_dates=["admittime", "dischtime"]
    )
    patients = pd.read_csv(HOSP_DIR / "patients.csv.gz")
    diagnoses = pd.read_csv(HOSP_DIR / "diagnoses_icd.csv.gz")
    transfers = pd.read_csv(
        HOSP_DIR / "transfers.csv.gz", parse_dates=["intime", "outtime"]
    )
    return admissions, patients, diagnoses, transfers


def assign_cohorts(admissions):
    """Label each admission emergency/elective and drop the excluded types."""
    adm = admissions.copy()
    adm["cohort"] = pd.NA
    adm.loc[adm["admission_type"].isin(EMERGENCY_TYPES), "cohort"] = "emergency"
    adm.loc[adm["admission_type"].isin(ELECTIVE_TYPES), "cohort"] = "elective"
    adm["in_strict_cohort"] = adm["admission_type"].isin(
        STRICT_EMERGENCY_TYPES + STRICT_ELECTIVE_TYPES
    )
    return adm.dropna(subset=["cohort"])


def build_unit_stays(transfers, cohort_adm):
    """Ordered care-unit stays per admission.

    - drops 'discharge' rows (they mark leaving the hospital and have no care unit)
    - drops rows with no hadm_id (ED visits that never became an admission)
    - keeps only admissions in the study cohorts
    - merges back-to-back rows in the same unit into one stay, so a bed change
      inside a unit is not counted as a transfer (this removes self-loops)
    """
    t = transfers[transfers["eventtype"] != "discharge"]
    t = t.dropna(subset=["hadm_id"])
    t = t.astype({"hadm_id": "int64"})
    t = t.merge(cohort_adm[["hadm_id", "cohort"]], on="hadm_id", how="inner")
    t = t.sort_values(["hadm_id", "intime"])

    # A new stay starts whenever the unit differs from the previous row.
    new_stay = t["careunit"] != t.groupby("hadm_id")["careunit"].shift()
    t["stay_id"] = new_stay.groupby(t["hadm_id"]).cumsum()

    stays = (
        t.groupby(["subject_id", "hadm_id", "cohort", "stay_id"], as_index=False)
        .agg(
            careunit=("careunit", "first"),
            eventtype=("eventtype", "first"),
            intime=("intime", "min"),
            outtime=("outtime", "max"),
        )
        .rename(columns={"stay_id": "stay_seq"})
    )
    stays["hours_in_unit"] = (
        (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600
    ).round(2)
    return stays


def build_transitions(stays):
    """Consecutive unit-to-unit moves within each admission (network edges)."""
    s = stays.sort_values(["hadm_id", "stay_seq"]).copy()
    s["to_unit"] = s.groupby("hadm_id")["careunit"].shift(-1)
    s["transition_time"] = s.groupby("hadm_id")["intime"].shift(-1)
    edges = s.dropna(subset=["to_unit"]).rename(columns={"careunit": "from_unit"})
    edges = edges[
        ~edges["from_unit"].isin(NON_UNITS) & ~edges["to_unit"].isin(NON_UNITS)
    ]
    return edges[
        ["subject_id", "hadm_id", "cohort", "from_unit", "to_unit", "transition_time"]
    ].reset_index(drop=True)


def build_admissions_table(cohort_adm, patients, diagnoses, stays):
    """One row per admission with the descriptive variables from the proposal."""
    adm = cohort_adm.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
    )
    # MIMIC shifts dates; age at admission = anchor_age + years since anchor_year
    adm["age"] = adm["anchor_age"] + (adm["admittime"].dt.year - adm["anchor_year"])
    adm["los_days"] = (
        (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
    ).round(2)

    n_dx = diagnoses.groupby("hadm_id").size().rename("n_diagnoses")
    path = stays.groupby("hadm_id").agg(
        path_length=("stay_seq", "size"),          # number of unit stays
        n_distinct_units=("careunit", "nunique"),
        first_unit=("careunit", "first"),
    )
    adm = adm.merge(n_dx, on="hadm_id", how="left").merge(path, on="hadm_id", how="left")
    adm["n_diagnoses"] = adm["n_diagnoses"].fillna(0).astype(int)
    adm["n_transitions"] = adm["path_length"] - 1
    adm["discharge_location"] = adm["discharge_location"].fillna("UNKNOWN")

    cols = [
        "subject_id", "hadm_id", "cohort", "admission_type", "in_strict_cohort",
        "admittime", "dischtime", "los_days", "admission_location",
        "discharge_location", "hospital_expire_flag", "gender", "age",
        "n_diagnoses", "path_length", "n_transitions", "n_distinct_units",
        "first_unit",
    ]
    return adm[cols].sort_values(["cohort", "subject_id", "admittime"])


def main():
    admissions, patients, diagnoses, transfers = load_raw()
    cohort_adm = assign_cohorts(admissions)
    stays = build_unit_stays(transfers, cohort_adm)
    edges = build_transitions(stays)
    adm = build_admissions_table(cohort_adm, patients, diagnoses, stays)

    adm.to_csv(DATA_DIR / "admissions_clean.csv", index=False)
    stays.to_csv(DATA_DIR / "unit_stays.csv", index=False)
    edges.to_csv(DATA_DIR / "transitions.csv", index=False)

    # ---- cleaning summary -------------------------------------------------
    print(f"Raw admissions:            {len(admissions)}")
    print(f"Excluded (observation):    {len(admissions) - len(cohort_adm)}")
    print(f"Study admissions:          {len(adm)}")
    print(adm.groupby("cohort").agg(admissions=("hadm_id", "size"),
                                    patients=("subject_id", "nunique")))
    both = adm.groupby("subject_id")["cohort"].nunique()
    print(f"Patients with admissions in both cohorts: {(both > 1).sum()}")
    print(f"Raw transfer rows:         {len(transfers)}")
    print(f"Unit stays kept:           {len(stays)}")
    print(f"Transitions (edges):       {len(edges)}")
    print(edges.groupby("cohort").size().rename("transitions"))
    print(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
