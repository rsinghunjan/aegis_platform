#!/usr/bin/env python3
"""
Common utilities for Aegis agents: Prometheus query, Argo submit/monitor, GitHub PR/Issue helpers,
Kubernetes patch helpers (for VirtualService weight adjustments).

Configure behavior via environment variables:
- PROMETHEUS_URL
- ARGO_SERVER (e.g., https://argo-server.namespace.svc.cluster.local)
- GITHUB_TOKEN
- GITHUB_REPO (owner/repo)
- KUBECONFIG (optional; in-cluster config used when running in K8s)
"""
import os
import time
import json
import requests
from urllib.parse import urljoin
from datetime import datetime, timedelta

# GitHub helper
try:
    from github import Github
except Exception:
    Github = None

# Kubernetes client
try:
    from kubernetes import client as k8s_client, config as k8s_config
    k8s_config.load_incluster_config()
    k8s_api = k8s_client.CustomObjectsApi()
    k8s_core = k8s_client.CoreV1Api()
    k8s_apps = k8s_client.AppsV1Api()
except Exception:
    k8s_client = None
    k8s_api = None
    k8s_core = None
    k8s_apps = None

PROM = os.environ.get("PROMETHEUS_URL", "http://prometheus.monitoring.svc.cluster.local:9090")
ARGO = os.environ.get("ARGO_SERVER", "http://argo-server.argo-workflows.svc.cluster.local:2746")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")  # e.g. owner/repo

def query_prometheus(query, timeout=30):
    """
    Run an instant Prometheus query and return JSON result.
    """
    url = urljoin(PROM, "/api/v1/query")
    resp = requests.get(url, params={"query": query}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError("Prometheus query failed: %s" % data)
    return data["data"]["result"]

def range_query_prometheus(query, start_seconds=300, step="30s"):
    """
    Run a range query for the last start_seconds.
    """
    end = datetime.utcnow()
    start = end - timedelta(seconds=start_seconds)
    url = urljoin(PROM, "/api/v1/query_range")
    params = {"query": query, "start": start.isoformat() + "Z", "end": end.isoformat() + "Z", "step": step}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError("Prometheus range query failed: %s" % data)
    return data["data"]["result"]

def argo_submit_workflow(workflow_manifest: dict):
    """
    Submit an Argo Workflow via the Argo server REST API.
    Returns the workflow name / metadata.
    """
    url = urljoin(ARGO, "/api/v1/workflows/aegis-ml")
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=workflow_manifest)
    resp.raise_for_status()
    return resp.json()

def argo_get_workflow(name):
    url = urljoin(ARGO, f"/api/v1/workflows/aegis-ml/{name}")
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def wait_for_argo_workflow(name, timeout_minutes=60, poll_seconds=10):
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        wf = argo_get_workflow(name)
        phase = wf.get("status", {}).get("phase", "")
        if phase in ("Succeeded", "Failed", "Error", "Running", "Pending"):
            if phase == "Succeeded":
                return wf
            if phase in ("Failed", "Error"):
                raise RuntimeError(f"Workflow {name} failed: {phase}")
        time.sleep(poll_seconds)
    raise TimeoutError("Timed out waiting for workflow %s" % name)

def create_github_pr(title, body, head_branch, base_branch="main", reviewers=None, labels=None):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        raise RuntimeError("GITHUB_TOKEN and GITHUB_REPO must be set for GitHub operations")
    if Github is None:
        raise RuntimeError("PyGithub not installed in agent image")
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)
    pr = repo.create_pull(title=title, body=body, head=head_branch, base=base_branch)
    if reviewers:
        try:
            pr.create_review_request(reviewers=reviewers)
        except Exception:
            pass
    if labels:
        try:
            pr.add_to_labels(*labels)
        except Exception:
            pass
    return pr.html_url

def create_github_issue(title, body, assignees=None, labels=None):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        raise RuntimeError("GITHUB_TOKEN and GITHUB_REPO must be set for GitHub operations")
    if Github is None:
        raise RuntimeError("PyGithub not installed in agent image")
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)
    issue = repo.create_issue(title=title, body=body, assignees=assignees or [], labels=labels or [])
    return issue.html_url

def patch_istio_virtualservice(namespace, name, new_spec):
    """
    Patch an Istio VirtualService using k8s API. new_spec is a patch (dict) applied to the virtualservice.
    Caution: This requires access to the Istio CRD api group.
    """
    if k8s_api is None:
        raise RuntimeError("Kubernetes client not available (are you running in-cluster?)")
    group = "networking.istio.io"
    version = "v1beta1"
    plural = "virtualservices"
    return k8s_api.patch_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural, name=name, body=new_spec)

def record_decision(db_conn_info, record):
    """
    Simple append to a JSONL file or HTTP endpoint for decisions; db_conn_info can be:
     - a path to a file in env DECISION_LOG_PATH
     - a URL in env DECISION_LOG_URL (POST JSON)
    """
    path = os.environ.get("DECISION_LOG_PATH")
    url = os.environ.get("DECISION_LOG_URL")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    elif url:
        requests.post(url, json=record, timeout=5)
    else:
        # fallback to stdout for ephemeral logs
        print("DECISION_LOG:", json.dumps(record))
