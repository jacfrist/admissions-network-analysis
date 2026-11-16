# admissions-network-analysis

Comparing intrahospital patient-transfer networks (in-degree and out-degree) between
emergency and elective admissions, using the MIMIC-IV Clinical Database Demo (v2.2).

## Project structure

```
admissions-network-analysis/
├── mimic-iv-data/      # MIMIC-IV Clinical Database Demo v2.2 (ODbL license)
├── data/               # cleaned / intermediate tables
├── src/                # reusable Python code (cleaning, cohorts, networks)
│   └── config.py       # shared file paths
├── notebooks/          # exploration and analysis notebooks
├── figures/            # plots for the report and slides
├── results/            # summary tables
└── requirements.txt
```

## Setup

The [MIMIC-IV Clinical Database Demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/)
is included in `mimic-iv-data/`. It is openly available under the Open Database License
(see `mimic-iv-data/LICENSE.txt`).

Install dependencies:
```
pip install -r requirements.txt
```

The raw data files are never modified; all cleaning happens in code.
