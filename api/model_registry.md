"""
Model registry utility for Aegis.

Responsibilities:
- Register a model + version into DB (Model + ModelVersion rows)
- Compute artifact checksum (SHA256) and HMAC signature (optional)
- Validate artifact via checksum
- Promote model versions (staging -> canary -> prod) by updating metadata JSON stored
- Expose simple prometheus counters for registrations/promotions

Notes:
- We store registry-specific fields inside ModelVersion.metadata (JSON) so we don't
  need a schema migration here; this keeps compatibility with the existing models schema.
- For production, you may want a dedicated table for lifecycle fields and signed attestations.
"""
import os
import hashlib
import hmac
import json
import logging
from typing import Optional, Dict, Any

from prometheus_client import Counter

from sqlalchemy.exc import IntegrityError
from api.db import SessionLocal
from api.models import Model, ModelVersion

logger = logging.getLogger("aegis_model_registry")
logging.basicConfig(level=logging.INFO)

# Prometheus metrics
REG_REGISTRATIONS = Counter("aegis_registry_registrations_total", "Total model registrations", ["status"])
REG_PROMOTIONS = Counter("aegis_registry_promotions_total", "Total model promotions", ["from_stage", "to_stage"])

# Signing key from env (HMAC). Must be configured in production.
MODEL_SIGNING_KEY = os.environ.get("AEGIS_MODEL_SIGN_KEY", "dev-model-sign-key-change-me")

# allowed lifecycle stages
ALLOWED_STAGES = ["staging", "canary", "prod"]


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hmac_sign(data: str) -> str:
    key = MODEL_SIGNING_KEY.encode("utf-8")
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).hexdigest()


def register_model_artifact(
    model_name: str,
    version: str,
    artifact_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    stage: str = "staging",
    validate_checksum: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Register or upsert a model + version in DB.
    - artifact_path may be local filesystem path or an object-store URL (s3://...)
    - If artifact_path is local, checksum will be computed.
    - The registry adds metadata fields:
        metadata['_registry'] = {
            'artifact_checksum': <sha256 or null>,
            'artifact_signature': <hmac>,
            'stage': <staging|canary|prod>,
            'validated': True/False,
        }
    Returns the model_version metadata dict.
    """
    session = SessionLocal()
    try:
        model = session.query(Model).filter_by(name=model_name).one_or_none()
        if model is None:
            model = Model(name=model_name, description=(metadata or {}).get("description"))
            session.add(model)
            session.flush()  # get model.id

        # compute checksum if artifact_path is local file
        checksum = None
        if os.path.exists(artifact_path) and os.path.isfile(artifact_path):
            checksum = _sha256_of_file(artifact_path)
            if validate_checksum and checksum != validate_checksum:
                raise ValueError("Checksum mismatch: artifact does not match provided checksum")

        # sign payload: model_name|version|checksum
        sign_payload = f"{model_name}|{version}|{checksum or ''}"
        signature = _hmac_sign(sign_payload)

        # existing version?
        mv = session.query(ModelVersion).filter_by(model_id=model.id, version=version).one_or_none()
        meta = metadata.copy() if metadata else {}
        registry_meta = {
            "artifact_checksum": checksum,
            "artifact_signature": signature,
            "stage": stage,
            "validated": bool(checksum is not None),
        }
        meta["_registry"] = registry_meta

        if mv is None:
            mv = ModelVersion(
                model_id=model.id,
                version=version,
                metadata=meta,
                artifact_path=artifact_path,
            )
            session.add(mv)
        else:
            mv.metadata = meta
            mv.artifact_path = artifact_path

        session.commit()

        REG_REGISTRATIONS.labels(status="success").inc()
        logger.info("Registered model %s:%s (stage=%s)", model_name, version, stage)
        return {"model": model_name, "version": version, "metadata": mv.metadata}
    except IntegrityError as e:
        session.rollback()
        REG_REGISTRATIONS.labels(status="error").inc()
        logger.exception("DB integrity error registering model: %s", e)
        raise
    except Exception:
        session.rollback()
        REG_REGISTRATIONS.labels(status="error").inc()
        logger.exception("Failed to register model %s:%s", model_name, version)
        raise
    finally:
        session.close()


def promote_model_version(model_name: str, version: str, target_stage: str, actor: Optional[str] = None) -> Dict[str, Any]:
    """
    Promote a model version from its current stage to target_stage.
    Allowed transitions are flexible, but this function enforces valid stages.
    Optionally attaches an audit field in metadata: _registry['promotion_history'] list.
    """
    if target_stage not in ALLOWED_STAGES:
        raise ValueError("invalid target stage")

    session = SessionLocal()
    try:
        model = session.query(Model).filter_by(name=model_name).one_or_none()
        if model is None:
            raise KeyError("model not found")
        mv = session.query(ModelVersion).filter_by(model_id=model.id, version=version).one_or_none()
        if mv is None:
            raise KeyError("model version not found")

        curr_stage = mv.metadata.get("_registry", {}).get("stage", "staging")
        # record promotion history
        history = mv.metadata.get("_registry", {}).get("promotion_history", [])
        history = list(history)
        history.append({"from": curr_stage, "to": target_stage, "actor": actor, "ts": __now_iso()})
        # update registry meta
        mv.metadata.setdefault("_registry", {})
        mv.metadata["_registry"]["stage"] = target_stage
        mv.metadata["_registry"]["promotion_history"] = history

        session.commit()
        REG_PROMOTIONS.labels(from_stage=curr_stage, to_stage=target_stage).inc()
        logger.info("Promoted %s:%s %s -> %s", model_name, version, curr_stage, target_stage)
        return {"model": model_name, "version": version, "from": curr_stage, "to": target_stage}
    except Exception:
        session.rollback()
        logger.exception("Failed to promote %s:%s", model_name, version)
        raise
    finally:
        session.close()


def get_model_versions(model_name: str) -> Dict[str, Any]:
    session = SessionLocal()
    try:
        model = session.query(Model).filter_by(name=model_name).one_or_none()
        if model is None:
            raise KeyError("model not found")
        versions = session.query(ModelVersion).filter_by(model_id=model.id).all()
        out = []
        for v in versions:
            out.append({"version": v.version, "artifact_path": v.artifact_path, "metadata": v.metadata})
        return {"model": model_name, "versions": out}
    finally:
        session.close()


def validate_model_signature(model_name: str, version: str) -> bool:
    """
    Verify artifact signature using HMAC; recompute using stored checksum.
    Returns True if signature matches.
    """
    session = SessionLocal()
    try:
        model = session.query(Model).filter_by(name=model_name).one_or_none()
        if model is None:
            raise KeyError("model not found")
        mv = session.query(ModelVersion).filter_by(model_id=model.id, version=version).one_or_none()
        if mv is None:
            raise KeyError("model version not found")
        registry_meta = (mv.metadata or {}).get("_registry", {})
        checksum = registry_meta.get("artifact_checksum")
        signature = registry_meta.get("artifact_signature")
        expected_sig = _hmac_sign(f"{model_name}|{version}|{checksum or ''}")
        ok = hmac.compare_digest(signature or "", expected_sig)
        # update validated flag
        mv.metadata.setdefault("_registry", {})
        mv.metadata["_registry"]["validated"] = ok
        session.commit()
        return ok
    finally:
        session.close()


def __now_iso():
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"
