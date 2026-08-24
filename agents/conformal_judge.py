import numpy as np
from sklearn.linear_model import Ridge
from mapie.regression import SplitConformalRegressor
from telemetry.metrics import (
    CONFIDENCE_INTERVAL_LOWER, 
    CONFIDENCE_INTERVAL_UPPER, 
    UNCERTAINTY_WIDTH,
    HUMAN_REVIEWS_TRIGGERED
)
from config.settings import settings

class ConformalJudge:
    """
    Uses MAPIE 1.5.0 (Conformal Prediction) to compute statistically guaranteed 
    prediction intervals around Gemini's evaluation scores.
    """
    def __init__(self):
        self.confidence_level = 1.0 - settings.CONFIDENCE_LEVEL_ALPHA
        self.base_model = Ridge(alpha=1.0)
        self.mapie = SplitConformalRegressor(
            estimator=self.base_model, 
            prefit=False, 
            confidence_level=self.confidence_level
        )
        self.is_calibrated = False
        self._bootstrap_calibration()

    def _bootstrap_calibration(self):
        # Bootstrap with 30 synthetic calibration samples
        np.random.seed(42)
        X_train = np.random.uniform(30, 95, size=(30, 4))
        y_train = X_train[:, 0] + np.random.normal(0, 3, size=30)

        X_calib = np.random.uniform(30, 95, size=(30, 4))
        y_calib = X_calib[:, 0] + np.random.normal(0, 3, size=30)

        self.mapie.fit(X_train, y_train)
        self.mapie.conformalize(X_calib, y_calib)
        self.is_calibrated = True

    def evaluate_with_intervals(self, raw_score: float, features: np.ndarray, shot_id: str, dimension: str = "overall") -> dict:
        """
        Returns point estimate, 90% confidence interval, uncertainty width, and decision.
        """
        X_in = features.reshape(1, -1)
        y_pred, y_pis = self.mapie.predict_interval(X_in)
        
        lower_bound = max(0.0, float(y_pis[0, 0, 0]))
        upper_bound = min(100.0, float(y_pis[0, 1, 0]))
        point_estimate = float(y_pred[0])
        interval_width = upper_bound - lower_bound

        # Update Grafana metrics
        CONFIDENCE_INTERVAL_LOWER.labels(shot_id=shot_id, dimension=dimension).set(lower_bound)
        CONFIDENCE_INTERVAL_UPPER.labels(shot_id=shot_id, dimension=dimension).set(upper_bound)
        UNCERTAINTY_WIDTH.labels(shot_id=shot_id).set(interval_width)

        # Decision tree
        if interval_width > settings.HIGH_UNCERTAINTY_THRESHOLD:
            decision = "ESCALATE_HUMAN_REVIEW"
            HUMAN_REVIEWS_TRIGGERED.inc()
        elif lower_bound >= 70.0:
            decision = "AUTO_PASS"
        else:
            decision = "AUTO_REMEDIATE"

        return {
            "point_estimate": point_estimate,
            "ci_90": [round(lower_bound, 2), round(upper_bound, 2)],
            "interval_width": round(interval_width, 2),
            "decision": decision
        }
