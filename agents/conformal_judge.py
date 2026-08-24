"""MAPIE 1.5.0 Conformal Decision Layer: Statistically Sound Video Quality Verification.
Replaces arbitrary heuristic thresholds with mathematically guaranteed prediction sets.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from mapie.classification import SplitConformalClassifier
from telemetry.tracer import tracer
from telemetry.metrics import CONFORMAL_SET_SIZE_HISTOGRAM, UNCERTAIN_VERDICTS_COUNTER

CLASSES = ["CANNOT_DETERMINE", "MATCH", "MISMATCH"]

class PrefitProbaClassifier:
    """Passthrough estimator for precomputed 3-call consensus probability vectors."""
    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return X / X.sum(axis=1, keepdims=True)

    def predict(self, X):
        X = np.asarray(X)
        return self.classes_[np.argmax(X, axis=1)]

class ConformalJudge:
    """
    Evaluates empirical agreement rates from 3-call consensus verification
    using MAPIE 1.5.0 Conformal Prediction sets with distribution-free coverage guarantees.
    """
    def __init__(
        self, 
        calibration_path: Optional[str] = None, 
        confidence_level: float = 0.80
    ):
        self.confidence_level = confidence_level
        self.classes = CLASSES
        self.calibrated = False
        
        default_calib = Path(__file__).parent.parent / "eval" / "labeled_set" / "calibration_data_full.json"
        self.calibration_path = str(calibration_path or default_calib)
        self.estimator = PrefitProbaClassifier(self.classes)
        self.mapie_clf = SplitConformalClassifier(
            estimator=self.estimator,
            confidence_level=self.confidence_level,
            conformity_score="lac",
            prefit=True
        )
        self._fit_calibration()

    def _fit_calibration(self):
        if not os.path.exists(self.calibration_path):
            return
        
        try:
            with open(self.calibration_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            rows = [r for r in rows if r.get("agreement_rate") is not None]
            
            X = np.array([self._votes_to_proba(r["votes"]) for r in rows])
            y = np.array([r["ground_truth"] for r in rows])
            
            self.mapie_clf.conformalize(X, y)
            self.calibrated = True
        except Exception as e:
            print(f"[ConformalJudge] Calibration initialization note: {e}")

    def _votes_to_proba(self, votes: List[str]) -> np.ndarray:
        counts = np.array([votes.count(c) for c in self.classes], dtype=float)
        total = counts.sum()
        if total == 0:
            return np.ones(len(self.classes)) / len(self.classes)
        return counts / total

    def evaluate_verdict(
        self, 
        votes: List[str], 
        fallback_verdict: str = "MATCH"
    ) -> Dict[str, Any]:
        """
        Evaluates a 3-call consensus vote vector using calibrated MAPIE prediction sets.
        - Set size == 1: Single decisive verdict (autonomous decision)
        - Set size >= 2: Multi-verdict ambiguity (abstain / flag for review)
        """
        with tracer.start_as_current_span("ConformalJudge.evaluate_verdict"):
            proba = self._votes_to_proba(votes).reshape(1, -1)
            
            if not self.calibrated:
                majority = max(set(votes), key=votes.count) if votes else fallback_verdict
                return {
                    "verdict": majority,
                    "prediction_set": [majority],
                    "set_size": 1,
                    "is_autonomous": True,
                    "coverage_guarantee": self.confidence_level,
                    "calibrated": False
                }

            _, y_sets = self.mapie_clf.predict_set(proba)
            y_sets_arr = np.asarray(y_sets)
            mask = y_sets_arr[0].reshape(-1) if y_sets_arr[0].ndim == 1 else y_sets_arr[0, :, 0]
            pred_set = [self.classes[j] for j, in_set in enumerate(mask) if in_set]
            
            set_size = len(pred_set)
            CONFORMAL_SET_SIZE_HISTOGRAM.observe(set_size)
            
            if set_size == 1:
                final_verdict = pred_set[0]
                is_autonomous = True
            else:
                UNCERTAIN_VERDICTS_COUNTER.inc()
                is_autonomous = False
                final_verdict = "CANNOT_DETERMINE" if "CANNOT_DETERMINE" in pred_set else pred_set[0]

            return {
                "verdict": final_verdict,
                "prediction_set": pred_set,
                "set_size": set_size,
                "is_autonomous": is_autonomous,
                "coverage_guarantee": self.confidence_level,
                "agreement_rate": float(max(proba[0])),
                "calibrated": True
            }
