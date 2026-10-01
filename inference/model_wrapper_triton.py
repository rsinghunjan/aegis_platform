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
 50
 51
 52
"""
Optional ModelWrapper that forwards inference requests to a Triton Inference Server (HTTP).

This is a minimal example. For production, use Triton's python client or gRPC for performance.
Assumes Triton model expects inputs named "INPUT__0" and returns JSON outputs.

Environment:
- TRITON_URL e.g. http://triton-server:8000
- TRITON_MODEL_NAME e.g. my_model
"""
from __future__ import annotations

import json
import logging
import os
from typing import List

import requests

logger = logging.getLogger(__name__)


class TritonModelWrapper:
    def __init__(self, model_name: str = None, triton_url: str = None, model_version: str = "1"):
        self.model_name = model_name or os.getenv("TRITON_MODEL_NAME", None)
        self.triton_url = triton_url or os.getenv("TRITON_URL", "http://localhost:8000")
        self.model_version = model_version

    def predict(self, texts: List[str]) -> List:
        """
        Simple HTTP request to Triton /v2/models/<model>/infer endpoint with JSON body.
        This minimal mapping expects model to accept a JSON array input and return JSON array outputs.
        Adjust to your Triton model input/output names & shapes.
        """
        if not self.model_name:
            raise RuntimeError("Triton model name is not configured")
        url = f"{self.triton_url}/v2/models/{self.model_name}/infer"
        # Minimal payload depending on model specifics. This is illustrative.
        payload = {
            "inputs": [
                {"name": "TEXT", "shape": [len(texts)], "datatype": "BYTES", "data": [t.encode("utf-8").decode("latin-1") for t in texts]}
            ]
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        out = resp.json()
        # Adapt parsing to your Triton model's output structure
        outputs = out.get("outputs", [])
        if outputs:
            # assume first output contains list of strings
            return outputs[0].get("data", [])
        return [None] * len(texts)
aegis_multimodal_ai_system/inference/model_wrapper_triton.py
