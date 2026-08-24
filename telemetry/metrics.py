from prometheus_client import Counter, Gauge, Histogram, Summary

# --- Quality & Alignment Gauges ---
PROMPT_ALIGNMENT_SCORE = Gauge(
    'cine_prompt_alignment_score', 
    'Prompt adherence score from Gemini (0-100)', 
    ['shot_id', 'dimension']
)

CONFIDENCE_INTERVAL_LOWER = Gauge(
    'cine_score_ci_lower_bound', 
    'MAPIE 90% Confidence Interval Lower Bound', 
    ['shot_id', 'dimension']
)

CONFIDENCE_INTERVAL_UPPER = Gauge(
    'cine_score_ci_upper_bound', 
    'MAPIE 90% Confidence Interval Upper Bound', 
    ['shot_id', 'dimension']
)

UNCERTAINTY_WIDTH = Gauge(
    'cine_uncertainty_interval_width', 
    'Width of MAPIE Confidence Interval (Uncertainty Index)', 
    ['shot_id']
)

# --- Production & Operations Counters ---
TAKES_TOTAL = Counter(
    'cine_takes_total', 
    'Total number of generated takes evaluated', 
    ['status', 'defect_type']
)

HUMAN_REVIEWS_TRIGGERED = Counter(
    'cine_human_reviews_triggered_total', 
    'Takes escalated to director review due to high uncertainty'
)

DOLLARS_SAVED_ESTIMATE = Counter(
    'cine_dollars_saved_total', 
    'Estimated GPU / API dollars saved via targeted prompt remediation'
)

# --- Latency & Performance Histograms ---
INSPECTION_DURATION_SECONDS = Histogram(
    'cine_inspection_duration_seconds', 
    'Time spent inspecting and evaluating video alignment',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

REMEDIATION_DURATION_SECONDS = Histogram(
    'cine_remediation_duration_seconds', 
    'Time spent generating prompt remediation plan'
)
