"""Quick artifact verification script."""
import json, sys
from pathlib import Path

root = Path('results/fl_xgboost')
checks = [
    ('models/fl_global_model_final.json', 'Final FL model'),
    ('metrics/fl_test_metrics.json', 'Test metrics'),
    ('security/aes_hmac_demonstration.json', 'AES/HMAC demo'),
    ('security/tls_demonstration.json', 'TLS demo'),
    ('security/full_security_evidence.json', 'Full security evidence'),
    ('unknown_attack/iso_test_results.json', 'IF test results'),
    ('unknown_attack/iso_ood_simulation.json', 'IF OOD simulation'),
    ('unknown_attack/iso_threshold.json', 'IF threshold'),
    ('rounds/all_rounds_summary.json', 'Round summary'),
    ('plots/fl_accuracy_f1_per_round.png', 'Accuracy plot'),
    ('plots/fl_confusion_matrix_top10.png', 'Confusion matrix plot'),
    ('plots/iso_anomaly_score_distribution.png', 'IF score plot'),
    ('resources/resource_usage.json', 'Resource usage'),
    ('reports/RESEARCH_REPORT.md', 'Research report'),
    ('logs/fl_experiment.log', 'Training log'),
]

print('=== TrustGuard-FL Artifact Verification ===')
all_ok = True
for rel, label in checks:
    p = root / rel
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    status = 'OK' if exists else 'MISSING'
    if not exists:
        all_ok = False
    if exists:
        print(f'  [{status}] {label}: {rel} ({size:,} bytes)')
    else:
        print(f'  [{status}] {label}: {rel}')

print()
print('=== Key Results ===')
try:
    m = json.loads((root / 'metrics/fl_test_metrics.json').read_text())
    print(f"  FL Test Accuracy: {m['accuracy']:.4f}")
    print(f"  FL Test Macro F1: {m['macro_f1']:.4f}")
except Exception as e:
    print(f"  [ERR] metrics: {e}")

try:
    iso = json.loads((root / 'unknown_attack/iso_test_results.json').read_text())
    print(f"  IF Unknown flagged: {iso['flagged_unknown_pct']:.1f}%")
    print(f"  IF XGBoost acc (in-dist): {iso['xgboost_accuracy_on_in_dist']:.4f}")
except Exception as e:
    print(f"  [ERR] iso_test: {e}")

try:
    ood = json.loads((root / 'unknown_attack/iso_ood_simulation.json').read_text())
    print(f"  IF OOD detection rate: {ood['ood_flagged_pct']:.0f}%")
except Exception as e:
    print(f"  [ERR] iso_ood: {e}")

try:
    sec = json.loads((root / 'security/aes_hmac_demonstration.json').read_text())
    print(f"  Tamper test: {sec['step_5_tamper_test']['hmac_result']}")
    print(f"  Update status: {sec['step_5_tamper_test']['update_status']}")
except Exception as e:
    print(f"  [ERR] security: {e}")

try:
    tls = json.loads((root / 'security/tls_demonstration.json').read_text())
    print(f"  TLS version: {tls['tls_version']}")
    print(f"  TLS cipher: {tls['tls_cipher']}")
except Exception as e:
    print(f"  [ERR] tls: {e}")

print()
print(f'All artifacts present: {all_ok}')
