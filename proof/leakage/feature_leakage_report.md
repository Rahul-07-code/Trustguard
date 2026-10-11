# Feature Leakage Audit

The model input schema contains 46 numeric flow/statistical features and excludes both target columns. No feature is named as a label or file identifier in the available schema. This cannot establish independence of upstream flow generation. Status: WARNING, not a confirmed leakage finding.
