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
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
#!/usr/bin/env python3
"""
Minimal Autopilot controller skeleton.

- Receives Alertmanager POSTs at /alerts
- Simple rule: if alert name contains "canary-failure" -> propose rollback
- Modes: audit (log), suggest (POST suggestion to governance UI), auto (execute helm rollback via subprocess)
- Produces auditable JSON decision record for each event in ./autopilot_decisions.db (append-only)
"""
from __future__ import annotations
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
import uvicorn
import httpx

APP = FastAPI()
DB_FILE = Path(os.environ.get("AUTOPILOT_DB", "./autopilot_decisions.db"))
MODE = os.environ.get("AUTOPILOT_MODE", "audit")  # audit | suggest | auto
GOV_URL = os.environ.get("AEGIS_GOVERNANCE_URL", "")  # POST suggestions here
HELM_NAMESPACE = os.environ.get("HELM_NAMESPACE", "aegis")
HELM_RELEASE = os.environ.get("HELM_RELEASE", "aegis-distribution")

def log_decision(decision: Dict[str, Any]):
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.utcnow().isoformat()+"Z", **decision}
    with DB_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print("Decision logged:", record)

async def post_suggestion(payload: Dict[str, Any]):
    if not GOV_URL:
        print("Governance URL not configured; skipping suggestion post")
        return
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(f"{GOV_URL}/autopilot/suggestion", json=payload)
            print("Posted suggestion, status:", resp.status_code)
        except Exception as e:
            print("Failed posting suggestion:", str(e))

def helm_rollback(release: str, namespace: str, revision: int | str = "0"):
    # revision "0" means rollback to previous release, adapt per your helm setup
    cmd = ["helm", "rollback", release, str(revision), "-n", namespace]
    print("Executing:", " ".join(cmd))
    return subprocess.run(cmd, check=False, capture_output=True, text=True)

@APP.post("/alerts")
async def alerts(req: Request):
    payload = await req.json()
    # Alertmanager v2 format: payload may contain 'alerts' list
    alerts = payload.get("alerts", []) if isinstance(payload, dict) else []
    for a in alerts:
        labels = a.get("labels", {})
        alertname = labels.get("alertname", "unknown")
        print("Received alert:", alertname)
        decision = {"alertname": alertname, "labels": labels, "action": None, "mode": MODE}
        # Simple rule: canary failure => propose/execute rollback
        if "canary-failure" in alertname or labels.get("severity") == "critical":
            decision["action"] = "rollback"
            decision["reason"] = "canary failure / critical alert"
            if MODE == "audit":
                # log only
                log_decision(decision)
            elif MODE == "suggest":
                # post suggestion to governance UI
                log_decision(decision)
                await post_suggestion(decision)
            elif MODE == "auto":
                # execute rollback (ensure helm is available & service account allowed)
                log_decision(decision)
                result = helm_rollback(HELM_RELEASE, HELM_NAMESPACE)
                decision["exec_result"] = {"rc": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
                log_decision(decision)
        else:
            # other alert types: log only for now
            decision["action"] = "noop"
            log_decision(decision)
    return {"received": len(alerts)}

if __name__ == "__main__":
    uvicorn.run(APP, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))
automation/autopilot/autopilot_service.py
