#!/usr/bin/env python3
"""
Autonomy KPI Prometheus exporter

Exposes /metrics in Prometheus format summarizing autonomy KPIs:
 - autonomy_percent (gauge)
 - ns_with_high_autonomy, total_namespaces (gauges)
 - recent_attestations_count (gauge)
 - pending_promotions_count (gauge)
 - circuit_state (0=closed,1=open)

Reads data from:
 - kube-system/aegis-autonomy-level ConfigMap
 - kube-system/aegis-autonomy-circuit ConfigMap
 - auto-promote audit dir (AUTOMATION_AUDIT_DIR)
 - Argo workflows (labels) for pending promotions

Run as a Deployment (manifest provided).
"""
import os
import time
import json
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge
from kubernetes import client, config

AUTONOMY_CM_NAME = os.environ.get("AUTONOMY_CM_NAME", "aegis-autonomy-level")
AUTONOMY_CM_NS = os.environ.get("AUTONOMY_CM_NS", "kube-system")
CIRCUIT_CM_NAME = os.environ.get("CIRCUIT_CM_NAME", "aegis-autonomy-circuit")
CIRCUIT_CM_NS = os.environ.get("CIRCUIT_CM_NS", "kube-system")
ARGO_NS = os.environ.get("ARGO_NS", "argo")
AUTOMATION_AUDIT_DIR = os.environ.get("AUTOMATION_AUDIT_DIR", "/var/lib/aegis/auto_promote_audit")
ATT_PREFIX = os.environ.get("ATTEST_PREFIX", "aegis-hil-attest")
PORT = int(os.environ.get("EXPORTER_PORT", "9112"))
WINDOW_HOURS = int(os.environ.get("WINDOW_HOURS", "24"))

g_autonomy_percent = Gauge('aegis_autonomy_percent', 'Percent of namespaces with autonomy >= supervised-plus')
g_ns_high = Gauge('aegis_ns_high_autonomy', 'Number of namespaces with high autonomy')
g_ns_total = Gauge('aegis_ns_total', 'Total namespaces with autonomy configured')
g_recent_attestations = Gauge('aegis_recent_attestations', 'Number of attestations in window')
g_pending_promotions = Gauge('aegis_pending_promotions', 'Pending auto-promote workflows count')
g_circuit_state = Gauge('aegis_circuit_open', 'Circuit breaker open=1 closed=0')

def k8s_client():
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    return client.CoreV1Api(), client.CustomObjectsApi()

def read_configmap(core, name, ns):
    try:
        cm = core.read_namespaced_config_map(name, ns)
        return cm.data or {}
    except Exception:
        return {}

def count_attestations(core, hours=WINDOW_HOURS):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    count = 0
    try:
        secrets = core.list_namespaced_secret("aegis-system")
        for s in secrets.items:
            if s.metadata.name.startswith(ATT_PREFIX):
                if s.metadata.creation_timestamp and s.metadata.creation_timestamp.replace(tzinfo=None) >= cutoff:
                    count += 1
    except Exception:
        # fallback: scan audit dir
        try:
            for fn in os.listdir(AUTOMATION_AUDIT_DIR) if os.path.exists(AUTOMATION_AUDIT_DIR) else []:
                path = os.path.join(AUTOMATION_AUDIT_DIR, fn)
                if os.path.getmtime(path) >= (time.time() - hours*3600):
                    count += 1
        except Exception:
            pass
    return count

def count_pending_promotions(api):
    try:
        res = api.list_namespaced_custom_object(group="argoproj.io", version="v1alpha1", namespace=ARGO_NS, plural="workflows")
        items = res.get("items", [])
        pending = [w for w in items if w.get("metadata",{}).get("labels",{}).get("aegis/auto-promote")=="pending"]
        return len(pending)
    except Exception:
        return 0

def compute_and_set_metrics():
    core, api = k8s_client()
    levels = read_configmap(core, AUTONOMY_CM_NAME, AUTONOMY_CM_NS)
    total_ns = len(levels)
    high = sum(1 for v in levels.values() if v in ("supervised-plus","autonomous"))
    attest_count = count_attestations(core)
    pending = count_pending_promotions(api)
    circuit = read_configmap(core, CIRCUIT_CM_NAME, CIRCUIT_CM_NS).get("state","closed")
    autonomy_percent = (high / total_ns * 100.0) if total_ns > 0 else 0.0

    g_autonomy_percent.set(round(autonomy_percent,2))
    g_ns_high.set(high)
    g_ns_total.set(total_ns)
    g_recent_attestations.set(attest_count)
    g_pending_promotions.set(pending)
    g_circuit_state.set(1 if circuit == "open" else 0)

def main():
    start_http_server(PORT)
    print("Autonomy exporter listening on port", PORT)
    while True:
        try:
            compute_and_set_metrics()
        except Exception as e:
            print("exporter error:", e)
        time.sleep(int(os.environ.get("EXPORTER_INTERVAL_S","30")))

if __name__ == "__main__":
    main()
