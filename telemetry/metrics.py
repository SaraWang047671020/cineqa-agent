class DummyMetric:
    def __init__(self, *args, **kwargs): pass
    def inc(self, *args, **kwargs): pass
    def dec(self, *args, **kwargs): pass
    def observe(self, *args, **kwargs): pass
    def set(self, *args, **kwargs): pass
    def labels(self, *args, **kwargs): return self

Counter = DummyMetric
Gauge = DummyMetric
Histogram = DummyMetric
Summary = DummyMetric

PROMPT_ALIGNMENT_SCORE = DummyMetric()
CONFIDENCE_INTERVAL_LOWER = DummyMetric()
CONFIDENCE_INTERVAL_UPPER = DummyMetric()
UNCERTAINTY_WIDTH = DummyMetric()
CONFORMAL_SET_SIZE_HISTOGRAM = DummyMetric()
UNCERTAIN_VERDICTS_COUNTER = DummyMetric()
TAKES_TOTAL = DummyMetric()
HUMAN_REVIEWS_TRIGGERED = DummyMetric()
DOLLARS_SAVED_ESTIMATE = DummyMetric()
INSPECTION_DURATION_SECONDS = DummyMetric()
REMEDIATION_DURATION_SECONDS = DummyMetric()

import os
import json
from datetime import datetime

def init_bq_table(*args, **kwargs):
    # Hijacked to init ClickHouse
    try:
        from database.clickhouse_init import init_tables
        init_tables()
    except Exception as e:
        print(f"[ClickHouse] Table init FAILED: {type(e).__name__}: {e}")

def log_take_to_bq(project_id, dataset_name, table_name, take_data, pass_rate, avg_set_size):
    # We will ignore this legacy function and rewrite ingestion natively in app.py or a new helper.
    pass
