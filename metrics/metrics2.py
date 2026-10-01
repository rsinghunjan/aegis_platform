  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
# Extended metrics: add queue depth gauge and error counters for model signature failures
from prometheus_client import Counter, Histogram, start_http_server, Gauge

SAFETY_FLAG_COUNTER = Counter(
    "aegis_safety_flags_total",
    "Total number of times the safety checker flagged input",
    ["reason", "detected_by"],
)

SAFETY_LATENCY_HISTOGRAM = Histogram(
    "aegis_safety_latency_seconds",
    "Latency in seconds for the safety checker",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0),
)

INFERENCE_REQUESTS_TOTAL = Counter(
    "aegis_inference_requests_total",
    "Total number of inference requests received",
    ["model_version"],
)

INFERENCE_PREDICTIONS_TOTAL = Counter(
    "aegis_inference_predictions_total",
    "Total number of output predictions generated",
    ["model_version"],
)

INFERENCE_LATENCY_HISTOGRAM = Histogram(
    "aegis_inference_latency_seconds",
    "Latency in seconds for the inference request (end-to-end)",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0),
)

# New: queue depth gauge (for autoscaling via prometheus-adapter)
AEGIS_INFERENCE_QUEUE_DEPTH = Gauge(
    "aegis_inference_queue_depth",
    "Current size of the inference batching queue (per process)",
)

# New: model signature / registry errors
AEGIS_MODEL_SIGNATURE_ERRORS = Counter(
    "aegis_model_signature_errors_total",
    "Total number of model signature verification failures",
    ["model_name", "model_version"],
)


def start_metrics_server(port: int = 8000) -> None:
    start_http_server(port)
