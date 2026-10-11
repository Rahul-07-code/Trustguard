# CICIoT23 21-Classes Dataset Directory

This directory stores the prepared CICIoT2023 21-class dataset splits used by TrustGuard-FL:

- `train_21.csv` (~1.59 GB, 5,392,858 rows)
- `validation_21.csv` (~342 MB, 1,155,488 rows)
- `test_21.csv` (~342 MB, 1,155,597 rows)
- `label_encoder_21.pkl` (LabelEncoder pickle mapping classes)

> [!NOTE]
> Large CSV data files (>100MB) are excluded from Git version control via `.gitignore`.
> Place the raw/prepared CSV files into this folder before running full local pipeline benchmarks.
