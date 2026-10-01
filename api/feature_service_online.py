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
"""
Snippet to fetch online features in a serving endpoint.

Integrate this into your inference runtime (FastAPI/Flask) to enrich model inputs
with features from Feast online store (Redis).
"""
from typing import Dict, Any
import os

from feast import FeatureStore

# Initialize feature store (only once at process startup)
FS_REPO = os.path.join("feast", "feature_repo")
fs = FeatureStore(repo_path=FS_REPO)

def enrich_request_with_features(entity_id: int) -> Dict[str, Any]:
    """
    Given a user/entity id, fetch online features and return a dict of feature values.
    """
    entity_rows = [{"user_id": int(entity_id)}]
    feature_refs = [
        "user_features:avg_session_length",
        "user_features:last_purchase_amount",
        "user_features:purchase_count_30d",
    ]
    # get_online_features returns an object with to_dict() method
    result = fs.get_online_features(features=feature_refs, entity_rows=entity_rows)
    rows = result.to_dict()
    # rows is a dict of feature_ref -> list of values (one per entity row)
    # convert to simple dict per-entity for first row:
    if not rows:
        return {}
    # map to simple name->value
    out = {}
    for k, v in rows.items():
        # k is like "user_features:avg_session_length"
        short = k.split(":")[-1]
        out[short] = v[0] if isinstance(v, list) and v else None
    return out

# Example usage inside a FastAPI handler:
# from fastapi import FastAPI
# app = FastAPI()
# @app.get("/predict/{user_id}")
# async def predict(user_id: int):
#     features = enrich_request_with_features(user_id)
#     # merge with request inputs and call model runner
#     return {"features": features}
