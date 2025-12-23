#!/usr/bin/env python3
"""
Query MLflow for the best run in a given experiment based on a specified metric.

Usage:
  python scripts/select_best_mlflow_run.py --experiment-name aegis-demo --metric val/accuracy --maximize
  python scripts/select_best_mlflow_run.py --experiment-name aegis-demo --metric val/loss --minimize

Outputs JSON with best run info including run_id, metrics, and artifact location.
Requires MLFLOW_TRACKING_URI environment variable or defaults to localhost:5000.
"""
import argparse
import json
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the best MLflow run from an experiment based on a metric"
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Name of the MLflow experiment to query",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help="Metric to use for selection (e.g., val/accuracy, val/loss)",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Select the run with the highest metric value",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Select the run with the lowest metric value",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path to write JSON output to file",
    )
    return parser.parse_args()


def select_best_run(experiment_name: str, metric: str, maximize: bool) -> dict:
    """
    Query MLflow for the best run in the specified experiment.

    Args:
        experiment_name: Name of the experiment to query
        metric: Metric key to use for selection
        maximize: If True, select max value; if False, select min value

    Returns:
        Dictionary with run information
    """
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    client = MlflowClient()

    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Search runs in the experiment
    order_direction = "DESC" if maximize else "ASC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.`{metric}` {order_direction}"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = runs[0]

    result = {
        "run_id": best_run.info.run_id,
        "experiment_id": best_run.info.experiment_id,
        "experiment_name": experiment_name,
        "artifact_uri": best_run.info.artifact_uri,
        "status": best_run.info.status,
        "start_time": best_run.info.start_time,
        "end_time": best_run.info.end_time,
        "metrics": dict(best_run.data.metrics),
        "params": dict(best_run.data.params),
        "selection_metric": metric,
        "selection_mode": "maximize" if maximize else "minimize",
        "selected_metric_value": best_run.data.metrics.get(metric),
    }

    return result


def main():
    args = parse_args()

    # Validate that exactly one of maximize or minimize is specified
    if args.maximize and args.minimize:
        print(
            "Error: Cannot specify both --maximize and --minimize",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.maximize and not args.minimize:
        print(
            "Error: Must specify one of --maximize or --minimize",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        result = select_best_run(
            experiment_name=args.experiment_name,
            metric=args.metric,
            maximize=args.maximize,
        )

        output_json = json.dumps(result, indent=2, default=str)
        print(output_json)

        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(output_json)
            print(f"Output written to {args.output_file}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
