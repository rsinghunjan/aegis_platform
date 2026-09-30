import logging
from typing import List, Optional

import flwr as fl
import numpy as np
from flwr.server.strategy import FedAvg

from metrics.metrics import start_metrics_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_config(server_round: int):
    return {"epoch_global": server_round}


def main():
    # Start metrics server so Prometheus can scrape metrics from this container/app.
    start_metrics_server(port=8000)
    logger.info("Prometheus metrics server started on :8000")

    # Configure strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=None,
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
