import argparse
import logging
import os
from typing import List, Tuple

import flwr as fl
import numpy as np
from sklearn.linear_model import SGDClassifier

from .utils import make_toy_client_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _sklearn_get_weights(model: SGDClassifier) -> List[np.ndarray]:
    # coef_ shape: (n_classes, n_features) or (n_features,) for binary classification depending on scikit-learn
    coef = model.coef_.astype("float32")
    intercept = model.intercept_.astype("float32")
    return [coef, intercept]


def _sklearn_set_weights(model: SGDClassifier, weights: List[np.ndarray]) -> None:
    coef, intercept = weights
    model.coef_ = coef
    model.intercept_ = intercept


class SklearnNumPyClient(fl.client.NumPyClient):
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray):
        self.cid = cid
        self.X = X
        self.y = y
        # initialize a model with appropriate classes for partial_fit
        self.model = SGDClassifier(loss="log_loss", max_iter=1, tol=None)
        # perform an initial partial_fit to create coef_ shape
        self.model.partial_fit(self.X[:2], self.y[:2], classes=np.array([0, 1]))

    def get_parameters(
        self, config: dict = None
    ) -> List[np.ndarray]:
        return _sklearn_get_weights(self.model)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        # set incoming global parameters
        _sklearn_set_weights(self.model, parameters)
        # local training
        self.model.partial_fit(self.X, self.y)
        # return updated parameters and number of examples trained on
        return _sklearn_get_weights(self.model), len(self.X), {}

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        _sklearn_set_weights(self.model, parameters)
        loss = 0.0
        try:
            preds = self.model.predict(self.X)
            accuracy = float((preds == self.y).mean())
            # In scikit-learn there's no simple loss returned here; return 1-accuracy as a proxy
            loss = 1.0 - accuracy
            return float(loss), len(self.X), {"accuracy": accuracy}
        except Exception as e:
            logger.exception("Evaluation failed: %s", e)
            return float(loss), len(self.X), {"accuracy": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client id")
    parser.add_argument("--host", type=str, default="localhost:8080", help="Flower server address")
    args = parser.parse_args()

    # create local toy data (vary random_state by client id)
    X, y = make_toy_client_data(n_samples=200, n_features=20, random_state=args.cid)

    client = SklearnNumPyClient(cid=args.cid, X=X, y=y)
    fl.client.start_numpy_client(server_address=args.host, client=client)


if __name__ == "__main__":
    main()
