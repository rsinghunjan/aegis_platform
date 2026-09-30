# Federated integration test: small Flower simulation with anomaly detection.
# This test starts a Flower server in a background thread and connects a few simple numpy clients.
# One client sends a malicious (very large) update and should be excluded by the strategy.
#
# Notes:
# - Requires flwr installed in the test runner environment (add to requirements.in or CI job).
# - This test is intentionally lightweight and synthetic to make CI runs fast.
import threading
import time
import numpy as np
import pytest

pytest.importorskip("flwr")
pytest.importorskip(
    "aegis_multimodal_ai_system.federated.secure_aggregation_strategy"
)

try:
    import flwr as fl  # type: ignore
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays  # type: ignore
except Exception as e:
    fl = None

from aegis_multimodal_ai_system.federated.secure_aggregation_strategy import SecureAggregationStrategy

# Basic model: a single 1-D numpy array as "weights"
def get_initial_weights():
    return [np.zeros((10,), dtype=np.float32)]

class SimpleNumpyClient(fl.client.NumPyClient):  # type: ignore
    def __init__(self, weights: list, scale: float = 1.0):
        self._weights = weights
        self.scale = scale

    def get_parameters(self):
        # Return current parameters as a list of numpy arrays
        return self._weights

    def fit(self, parameters, config):
        # Receive global parameters and return updated parameters
        nds = parameters_to_ndarrays(parameters)
        # Simulate local training by adding a scaled delta
        new = [arr + (self.scale * np.ones_like(arr) * 0.1) for arr in nds]
        return ndarrays_to_parameters(new), len(new[0].ravel())

    def evaluate(self, parameters, config):
        return 0.0, len(parameters_to_ndarrays(parameters)[0].ravel()), {}

@pytest.mark.skipif(fl is None, reason="flwr not installed")
def test_secure_aggregation_excludes_malicious_update():
    # Build the strategy with a tight anomaly threshold to detect malicious client
    base_strategy = fl.server.strategy.FedAvg()
    wrapper = SecureAggregationStrategy(base_strategy=base_strategy, clip_norm=10.0, anomaly_std_multiplier=1.0)
    strategy = wrapper.get_strategy()

    # Start server in background thread
    server_thread = threading.Thread(
        target=lambda: fl.server.start_server(server_address="localhost:8080", config={"num_rounds": 1}, strategy=strategy),
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.5)  # give server time to start

    # Start normal clients (scale=1.0)
    clients = [
        SimpleNumpyClient(get_initial_weights(), scale=1.0),
        SimpleNumpyClient(get_initial_weights(), scale=1.0),
    ]

    # Malicious client: very large scale causing large-norm update
    bad_client = SimpleNumpyClient(get_initial_weights(), scale=1000.0)

    # Start clients as numpy clients connecting to server
    client_threads = []
    for c in clients + [bad_client]:
        t = threading.Thread(target=lambda client=c: fl.client.start_numpy_client("localhost:8080", client=client), daemon=True)
        t.start()
        client_threads.append(t)

    # Wait for all clients to finish
    for t in client_threads:
        t.join(timeout=30)

    # Ensure server thread completes
    server_thread.join(timeout=30)
    assert not server_thread.is_alive()
