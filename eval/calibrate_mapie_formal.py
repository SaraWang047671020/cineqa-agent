import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from mapie.classification import SplitConformalClassifier

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CLASSES = ["CANNOT_DETERMINE", "MATCH", "MISMATCH"]

class PrefitProbaClassifier:
    """Passthrough estimator for precomputed 3-call consensus probability vectors."""
    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.asarray(X)

    def predict(self, X):
        X = np.asarray(X)
        return self.classes_[np.argmax(X, axis=1)]

def votes_to_proba(votes: List[str]) -> np.ndarray:
    counts = np.array([votes.count(c) for c in CLASSES], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.ones(len(CLASSES)) / len(CLASSES)
    return counts / total

def run_multi_seed_analysis(
    data_path: str,
    target_levels: List[float] = [0.80, 0.85, 0.90],
    conformalize_ratio: float = 0.80,
    n_seeds: int = 50
) -> Dict[float, Dict[str, Any]]:
    """Runs Monte Carlo multi-seed cross validation across N random splits."""
    rows = json.loads(Path(data_path).read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("agreement_rate") is not None]
    
    X = np.array([votes_to_proba(r["votes"]) for r in rows])
    y = np.array([r["ground_truth"] for r in rows])
    n_samples = len(rows)

    results = {}
    for level in target_levels:
        coverages = []
        single_verdict_rates = []
        set_sizes = []
        thresholds = []

        split_size = int(n_samples * conformalize_ratio)

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n_samples)
            conf_idx, test_idx = idx[:split_size], idx[split_size:]

            X_conf, y_conf = X[conf_idx], y[conf_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            estimator = PrefitProbaClassifier(CLASSES)
            mapie_clf = SplitConformalClassifier(
                estimator=estimator,
                confidence_level=level,
                conformity_score="lac",
                prefit=True
            )
            mapie_clf.conformalize(X_conf, y_conf)

            # Prediction sets on test split
            _, y_sets = mapie_clf.predict_set(X_test)
            y_sets_arr = np.asarray(y_sets)

            # Evaluate test metrics
            covered = 0
            single_count = 0
            sizes = []
            for i, test_i in enumerate(test_idx):
                mask = y_sets_arr[i].reshape(-1) if y_sets_arr[i].ndim == 1 else y_sets_arr[i, :, 0]
                pred_set = [CLASSES[j] for j, in_set in enumerate(mask) if in_set]
                in_set = y_test[i] in pred_set
                covered += in_set
                single_count += (len(pred_set) == 1)
                sizes.append(len(pred_set))

            coverages.append(covered / len(test_idx))
            single_verdict_rates.append(single_count / len(test_idx))
            set_sizes.append(np.mean(sizes))

        results[level] = {
            "target_confidence": level,
            "empirical_coverage_mean": float(np.mean(coverages)),
            "empirical_coverage_std": float(np.std(coverages)),
            "single_verdict_rate_mean": float(np.mean(single_verdict_rates)),
            "single_verdict_rate_std": float(np.std(single_verdict_rates)),
            "average_set_size_mean": float(np.mean(set_sizes)),
            "n_samples": n_samples,
            "n_calib": split_size,
            "n_test": n_samples - split_size,
            "n_seeds": n_seeds
        }

    return results

def run_k_fold_conformal(
    data_path: str,
    target_levels: List[float] = [0.80, 0.85, 0.90],
    n_splits: int = 5
) -> Dict[float, Dict[str, Any]]:
    """Runs K-Fold Cross-Conformal Prediction utilizing 100% of data points."""
    rows = json.loads(Path(data_path).read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("agreement_rate") is not None]
    
    X = np.array([votes_to_proba(r["votes"]) for r in rows])
    y = np.array([r["ground_truth"] for r in rows])
    n_samples = len(rows)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    kfold_results = {}
    for level in target_levels:
        fold_coverages = []
        fold_single_rates = []
        fold_set_sizes = []

        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            estimator = PrefitProbaClassifier(CLASSES)
            mapie_clf = SplitConformalClassifier(
                estimator=estimator,
                confidence_level=level,
                conformity_score="lac",
                prefit=True
            )
            mapie_clf.conformalize(X_train, y_train)

            _, y_sets = mapie_clf.predict_set(X_val)
            y_sets_arr = np.asarray(y_sets)

            covered = 0
            single_count = 0
            sizes = []
            for i, val_i in enumerate(val_idx):
                mask = y_sets_arr[i].reshape(-1) if y_sets_arr[i].ndim == 1 else y_sets_arr[i, :, 0]
                pred_set = [CLASSES[j] for j, in_set in enumerate(mask) if in_set]
                in_set = y_val[i] in pred_set
                covered += in_set
                single_count += (len(pred_set) == 1)
                sizes.append(len(pred_set))

            fold_coverages.append(covered / len(val_idx))
            fold_single_rates.append(single_count / len(val_idx))
            fold_set_sizes.append(np.mean(sizes))

        kfold_results[level] = {
            "target_confidence": level,
            "kfold_coverage_mean": float(np.mean(fold_coverages)),
            "kfold_coverage_std": float(np.std(fold_coverages)),
            "kfold_single_verdict_mean": float(np.mean(fold_single_rates)),
            "kfold_avg_set_size": float(np.mean(fold_set_sizes)),
            "n_folds": n_splits,
            "total_samples": n_samples
        }

    return kfold_results

if __name__ == "__main__":
    data_file = r"C:\dev\hackathon\cineqa_agent\eval\labeled_set\calibration_data_full.json"
    print("=" * 80)
    print("🎯 MAPIE 1.5.0 Decision Layer Formal Calibration & Stability Analysis (92 Samples)")
    print("=" * 80)

    # 1. 50-Seed Monte Carlo Analysis (80% Conformalize / 20% Test)
    mc_results = run_multi_seed_analysis(data_file, target_levels=[0.80, 0.85, 0.90], conformalize_ratio=0.80, n_seeds=50)
    
    print("\n📊 1. Monte Carlo 50-Seed Stability Analysis (80% Calibration / 20% Holdout Test):")
    print("-" * 80)
    print(f"{'Target Conf':<12} | {'Empirical Coverage':<22} | {'Single Verdict Rate':<22} | {'Avg Set Size':<12}")
    print("-" * 80)
    for lvl, res in mc_results.items():
        cov_str = f"{res['empirical_coverage_mean']:.1%} ± {res['empirical_coverage_std']:.1%}"
        sing_str = f"{res['single_verdict_rate_mean']:.1%} ± {res['single_verdict_rate_std']:.1%}"
        print(f"{lvl:.0%:<12} | {cov_str:<22} | {sing_str:<22} | {res['average_set_size_mean']:.2f}")

    # 2. 5-Fold Cross-Conformal Full Ingestion
    kfold_results = run_k_fold_conformal(data_file, target_levels=[0.80, 0.85, 0.90], n_splits=5)
    print("\n📊 2. 5-Fold Cross-Conformal Analysis (100% Data Ingestion):")
    print("-" * 80)
    print(f"{'Target Conf':<12} | {'K-Fold Coverage':<22} | {'Single Verdict Rate':<22} | {'Avg Set Size':<12}")
    print("-" * 80)
    for lvl, res in kfold_results.items():
        cov_str = f"{res['kfold_coverage_mean']:.1%} ± {res['kfold_coverage_std']:.1%}"
        sing_str = f"{res['kfold_single_verdict_mean']:.1%}"
        print(f"{lvl:.0%:<12} | {cov_str:<22} | {sing_str:<22} | {res['kfold_avg_set_size']:.2f}")

    # Save results to json
    report_data = {
        "monte_carlo_50_seeds": mc_results,
        "kfold_cross_conformal": kfold_results
    }
    out_json = r"C:\dev\hackathon\cineqa_agent\eval\labeled_set\mapie_formal_calibration_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n✅ Metrics saved to {out_json}")
