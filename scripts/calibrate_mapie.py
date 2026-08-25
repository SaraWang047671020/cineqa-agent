import sys
import os
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
from mapie.classification import SplitConformalClassifier

CLASSES = ["CANNOT_DETERMINE", "MATCH", "MISMATCH"]

class PrefitProbaClassifier:
    """Pass-through estimator for MAPIE classification on consensus probabilities."""
    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.asarray(X)

    def predict(self, X):
        X = np.asarray(X)
        return self.classes_[np.argmax(X, axis=1)]

def votes_to_proba(votes):
    counts = np.array([votes.count(c) for c in CLASSES], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.ones(len(CLASSES)) / len(CLASSES)
    return counts / total

def run_calibration(data_path="eval/labeled_set/calibration_data_full.json", confidence_level=0.75):
    print("=" * 70)
    print(f"🎯 Calibrating MAPIE 1.5.0 Conformal Classifier (Confidence: {confidence_level:.0%})")
    print("=" * 70)

    full_path = os.path.join(root_dir, data_path)
    with open(full_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    # Filter rows with agreement rates
    valid_rows = [r for r in rows if r.get("agreement_rate") is not None and r.get("ground_truth") in CLASSES]
    print(f"Loaded {len(valid_rows)} benchmark rows with consensus ground truth.")

    X = np.array([votes_to_proba(r["votes"]) for r in valid_rows])
    y = np.array([r["ground_truth"] for r in valid_rows])

    # 60% calibration, 40% test split
    np.random.seed(42)
    indices = np.random.permutation(len(valid_rows))
    split = int(len(valid_rows) * 0.6)
    conf_idx, test_idx = indices[:split], indices[split:]

    X_conf, y_conf = X[conf_idx], y[conf_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    estimator = PrefitProbaClassifier(CLASSES)
    mapie_clf = SplitConformalClassifier(
        estimator=estimator,
        confidence_level=confidence_level,
        conformity_score="lac",
        prefit=True
    )

    mapie_clf.conformalize(X_conf, y_conf)
    y_pred, y_sets = mapie_clf.predict_set(X_test)

    y_sets_arr = np.asarray(y_sets)
    covered = 0
    single_decision_count = 0

    for i in range(len(test_idx)):
        mask = y_sets_arr[i].reshape(-1) if y_sets_arr[i].ndim == 1 else y_sets_arr[i, :, 0]
        pred_set = [CLASSES[j] for j, in_set in enumerate(mask) if in_set]
        gt = y_test[i]
        in_set = gt in pred_set
        covered += int(in_set)
        single_decision_count += int(len(pred_set) == 1)

    n_test = len(test_idx)
    emp_coverage = covered / n_test
    autonomous_ratio = single_decision_count / n_test

    print("-" * 70)
    print(f"✅ Empirical Coverage Rate : {emp_coverage:.1%} (Target Guaranteed: {confidence_level:.0%})")
    print(f"⚡ Autonomous Decision Rate : {autonomous_ratio:.1%} (Single Verdict without Escalation)")
    print(f"🛡️ Safety Escalation Rate   : {(1 - autonomous_ratio):.1%} (Abstained / Escalated to Director)")
    print("=" * 70)

if __name__ == "__main__":
    run_calibration()
