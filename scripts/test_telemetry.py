import sys
import os
import time
import urllib.request

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from prometheus_client import start_http_server
from telemetry.metrics import (
    PROMPT_ALIGNMENT_SCORE,
    CONFIDENCE_INTERVAL_LOWER,
    CONFIDENCE_INTERVAL_UPPER,
    UNCERTAINTY_WIDTH,
    TAKES_TOTAL,
    DOLLARS_SAVED_ESTIMATE
)
from telemetry.tracer import tracer

def test_telemetry():
    print("=" * 60)
    print("📊 Testing Prometheus Metrics & OpenTelemetry Pipeline")
    print("=" * 60)

    port = 8001
    start_http_server(port)
    print(f"Prometheus HTTP metrics server running on http://localhost:{port}")

    with tracer.start_as_current_span("test_span_telemetry_check"):
        PROMPT_ALIGNMENT_SCORE.labels(shot_id="shot_test", dimension="action").set(88.5)
        CONFIDENCE_INTERVAL_LOWER.labels(shot_id="shot_test", dimension="action").set(82.0)
        CONFIDENCE_INTERVAL_UPPER.labels(shot_id="shot_test", dimension="action").set(94.0)
        UNCERTAINTY_WIDTH.labels(shot_id="shot_test").set(12.0)
        TAKES_TOTAL.labels(status="passed", defect_type="none").inc()
        DOLLARS_SAVED_ESTIMATE.inc(0.35)

    time.sleep(0.5)
    url = f"http://localhost:{port}/metrics"
    req = urllib.request.urlopen(url)
    content = req.read().decode("utf-8")

    assert "cine_prompt_alignment_score" in content, "Missing alignment score metric"
    assert "cine_score_ci_lower_bound" in content, "Missing lower bound metric"
    assert "cine_dollars_saved_total" in content, "Missing dollars saved metric"

    print("\n[SUCCESS] Prometheus metrics scraping verified successfully!")
    print("Sample scraped metrics snippet:")
    for line in content.splitlines():
        if line.startswith("cine_") and not line.startswith("cine_inspection_duration"):
            print(f"  {line}")

if __name__ == "__main__":
    test_telemetry()
