# Data directory

Raw datasets and derived artifacts live here. **Nothing in `raw/` or
`processed/` is committed to git** (see `.gitignore`) — clinical corpora are
large and several carry licences that forbid redistribution.

```
data/
├── raw/          # downloaded datasets, exactly as published
└── processed/    # derived tables and splits
```

## Get the data

```bash
python scripts/download_data.py --all      # open corpora (~1 GB)
python scripts/download_data.py --list     # show the registry
```

This writes `data/processed/interactions.parquet` - the canonical interaction
table with the schema:

| Column | Type | Description |
|---|---|---|
| `drug_a` | str | normalised generic drug name (lower-case) |
| `drug_b` | str | normalised generic drug name (lower-case) |
| `event_type` | str | mechanism, e.g. `bleeding` |
| `severity` | str | `none`/`minor`/`moderate`/`major`/`contraindicated`/`unknown` |
| `description` | str | human-readable clinical consequence |
| `source` | str | which corpus the row came from |
| `source_id` | str | stable id used for citations |

Then build the splits:

```bash
python scripts/make_splits.py --strategy cold_start
python scripts/make_splits.py --compare      # produces the leakage table
```

## Prescription images

For your own OCR evaluation, collect images into `data/raw/prescriptions/`
with a consent note, and record provenance in a `manifest.csv`:

```csv
filename,consent_id,collected_on,source,notes
rx_001,C-2026-001,2026-11-05,clinic_A,handwritten
```

Never commit patient identifiers. Blur or redact the name, age, and
address before storing.
