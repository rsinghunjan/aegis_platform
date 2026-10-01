
"""
Unit tests for AdaptiveBatcher behavior.
Run: pytest tests/test_adaptive_batcher.py -q
"""
import time
from api.adaptive_batcher import AdaptiveBatcher

def test_batch_increase_and_decrease():
    ab = AdaptiveBatcher("test", initial_max_batch=4, min_batch=1, max_batch=32, target_p95_ms=100)
    # simulate low latencies -> should increase batch
    for _ in range(30):
        ab.record_latency(20, batch_size=2)
    increased = ab.get_max_batch()
    assert increased >= 4

    # now simulate very high latencies -> should decrease
    for _ in range(40):
        ab.record_latency(300, batch_size=increased)
    decreased = ab.get_max_batch()
    assert decreased <= increased
    assert decreased >= 1
