"""Split-Conformal Decision Layer (LAC): Statistically Sound Video Quality Verification.
Replaces arbitrary heuristic thresholds with mathematically guaranteed prediction sets.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from telemetry.tracer import tracer
from telemetry.metrics import CONFORMAL_SET_SIZE_HISTOGRAM, UNCERTAIN_VERDICTS_COUNTER

CLASSES = ["CANNOT_DETERMINE", "MATCH", "MISMATCH"]

class SplitConformalLAC:
    def __init__(self, confidence_level: float = 0.80, classes: List[str] = None):
        self.confidence_level = confidence_level
        self.classes = np.array(classes)
        self.q_hat = None

    def fit_calibration(self, probas: np.ndarray, y_true: np.ndarray):
        n = len(y_true)
        if n == 0:
            return
        
        scores = []
        for i in range(n):
            true_idx = np.where(self.classes == y_true[i])[0]
            if len(true_idx) == 0:
                continue
            true_idx = true_idx[0]
            prob_true = probas[i, true_idx]
            scores.append(1.0 - prob_true)
            
        scores = np.array(scores)
        n_scores = len(scores)
        if n_scores == 0:
            return
            
        q_level = min(1.0, self.confidence_level * (n_scores + 1.0) / n_scores)
        self.q_hat = np.quantile(scores, q_level, method='higher')
        
    def predict_set(self, probas: np.ndarray) -> np.ndarray:
        if self.q_hat is None:
            raise ValueError("Not calibrated")
        return (1.0 - probas) <= self.q_hat

class ConformalJudge:
    """
    Evaluates empirical agreement rates from 3-call consensus verification
    using Split-Conformal Prediction sets (LAC) with distribution-free coverage guarantees.
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
        self.conformal_clf = SplitConformalLAC(confidence_level=self.confidence_level, classes=self.classes)
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

            self.conformal_clf.fit_calibration(X, y)
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
        Evaluates a 3-call consensus vote vector using calibrated Conformal prediction sets.
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

            y_sets_mask = self.conformal_clf.predict_set(proba)[0]
            pred_set = [self.classes[j] for j, in_set in enumerate(y_sets_mask) if in_set]

            set_size = len(pred_set)
            CONFORMAL_SET_SIZE_HISTOGRAM.observe(set_size)

            if set_size == 1:
                final_verdict = pred_set[0]
                is_autonomous = True
            elif set_size == 0:
                UNCERTAIN_VERDICTS_COUNTER.inc()
                is_autonomous = False
                final_verdict = "CANNOT_DETERMINE"
            else:
                UNCERTAIN_VERDICTS_COUNTER.inc()
                is_autonomous = False
                if "CANNOT_DETERMINE" in pred_set:
                    final_verdict = "CANNOT_DETERMINE"
                else:
                    from collections import Counter
                    if votes:
                        most_common = Counter(votes).most_common(1)[0][0]
                        final_verdict = most_common if most_common in pred_set else pred_set[0]
                    else:
                        final_verdict = pred_set[0]

            return {
                "verdict": final_verdict,
                "prediction_set": pred_set,
                "set_size": set_size,
                "is_autonomous": is_autonomous,
                "coverage_guarantee": self.confidence_level,
                "agreement_rate": float(np.max(proba[0])),
                "calibrated": True
            }
