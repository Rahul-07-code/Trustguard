# Preprocessing Leakage Audit

The located training script performs no fitted scaler, imputer, encoder, PCA, feature-selection, aggregation, or target-derived transformation. It reads numeric CSV features directly and excludes `label` and `label_encoded`. The repository includes `label_encoder_21.pkl`, but the training script does not load it; its provenance is undocumented.

Status: PASS for the located training path; WARNING for undocumented upstream preparation.
