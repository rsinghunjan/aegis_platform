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
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
"""
Auto-canary controller: watch the model_versions table for promotions to 'canary'
and materialize a Kubernetes Deployment + Flagger Canary CR to initiate automated analysis.

- Run in-cluster (as a Deployment) or locally (with KUBECONFIG).
- Requires RBAC: permissions to create Deployments and Canary CRs in the target namespace.
- The controller assumes your API image can accept MODEL_NAME and MODEL_VERSION env vars and will load the specified model at startup.

How it works:
- Poll DB every N seconds for ModelVersion rows with metadata._registry.stage == 'canary' and not yet processed.
- For each candidate: create a Deployment (aegis-api-canary-<model>-<version>-<shortid>) and a Canary CR (flagger).
- Annotate ModelVersion.metadata._registry['canary_deployed'] with the k8s resource name (so we don't redeploy).
"""
import os
import time
import json
import logging
import hashlib
from typing import Optional

from kubernetes import client, config
from sqlalchemy import select
from api.db import SessionLocal
from api.models import Model, ModelVersion

logger = logging.getLogger("aegis.auto_canary")
logging.basicConfig(level=logging.INFO)

NAMESPACE = os.environ.get("CANARY_NAMESPACE", "default")
IMAGE = os.environ.get("AEGIS_IMAGE", "ghcr.io/your-org/aegis-api:latest")
POLL_INTERVAL = int(os.environ.get("CANARY_POLL_SEC", "15"))

# K8s API setup (in-cluster or via KUBECONFIG)
try:
    config.load_incluster_config()
except Exception:
    kubeconfig = os.environ.get("KUBECONFIG")
    if kubeconfig:
        config.load_kube_config(config_file=kubeconfig)
    else:
        config.load_kube_config()

