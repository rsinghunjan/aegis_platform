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
#!/usr/bin/env python3
"""
Lightweight governance UI: Flask app that lists model registry entries, shows model_signature.json,
displays evidence, and provides a demo inference playground that calls an orchestrator endpoint.

Run:
  pip install -r governance/requirements.txt
  python governance/ui_app.py

Config via env:
  MODEL_REGISTRY_DIR (default: model_registry)
  ORCH_URL (optional) — if provided, inference will call the orchestrator endpoint
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify

MODEL_REGISTRY_DIR = Path(os.environ.get("MODEL_REGISTRY_DIR", "model_registry"))
ORCH_URL = os.environ.get("ORCH_URL", "").rstrip("/")

app = Flask(__name__, template_folder=Path(__file__).parent / "templates")

def list_models():
    models = []
    for p in MODEL_REGISTRY_DIR.rglob("model_signature.json"):
        try:
            sig = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_dir = p.parent
        models.append({
            "name": model_dir.name,
            "path": str(model_dir),
            "signature": sig
        })
    return sorted(models, key=lambda x: x["path"])

@app.route("/")
def index():
    models = list_models()
    return render_template("index.html", models=models, orch_url=ORCH_URL)

@app.route("/model/<path:model_path>")
def model_detail(model_path):
    sig_path = Path(model_path) / "model_signature.json"
    if not sig_path.exists():
        return jsonify({"error": "model_signature.json not found"}), 404
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    return jsonify(sig)

@app.route("/infer", methods=["POST"])
def infer():
    payload = request.get_json(force=True)
    # Proxy to orchestrator if configured
    if ORCH_URL:
        try:
            resp = requests.post(ORCH_URL, json=payload, timeout=30)
            return (resp.content, resp.status_code, resp.headers.items())
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No ORCH_URL configured"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
