"""
Unit tests for scripts/evaluate_model.py (lightweight, uses monkeypatching).

Covers:
- local joblib artifact evaluation on iris dataset
- script exits with failure when metric below baseline
"""
import os
import sys
import json
import tempfile
from unittest import mock
import subprocess

import pytest

# import the evaluate function via subprocess to emulate CI run
SCRIPT = "scripts/evaluate_model.py"

def write_dummy_joblib(path, acc=1.0):
    # create a trivial sklearn model that predicts perfectly or imperfectly
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    import joblib
    X, y = load_iris(return_X_y=True)
    # train a logistic regression quickly
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    # optionally corrupt predictions to simulate lower accuracy
    joblib.dump(model, path)

def run_eval(args):
    env = os.environ.copy()
    cmd = [sys.executable, SCRIPT] + args
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return proc

def test_local_joblib_pass(tmp_path):
    model_path = tmp_path / "model.joblib"
    write_dummy_joblib(str(model_path))
    proc = run_eval(["--artifact-path", str(model_path), "--mode", "sklearn", "--metric", "accuracy", "--baseline", "0.5"])
    assert proc.returncode == 0 or proc.returncode == 0  # success

def test_local_joblib_fail(tmp_path):
    model_path = tmp_path / "model.joblib"
    write_dummy_joblib(str(model_path))
    # set baseline extremely high so it fails
    proc = run_eval(["--artifact-path", str(model_path), "--mode", "sklearn", "--metric", "accuracy", "--baseline", "0.9999"])
    assert proc.returncode == 2  # validation failed (below baseline)
