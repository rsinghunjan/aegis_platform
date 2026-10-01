"""
Canonical cache key generation for prompt/response caching.

Key includes:
- model_name@model_version
- system_prompt_hash (if any)
- canonicalized prompt text
- retrieval_fingerprint (if RAG used)
- sorted generation params (temperature, max_tokens, top_k/top_p, stop tokens)
"""
import hashlib
import json


def canonicalize_prompt(prompt: str) -> str:
    # simple canonicalization: normalize whitespace and newline, strip
    return " ".join(prompt.strip().split())


def fingerprint_retrieval(retrieval_meta):
    if not retrieval_meta:
        return ""
    # retrieval_meta expected to be list of retrieval items with doc_id + chunk_id + hash
    items = []
    for r in retrieval_meta:
        doc_id = r.get("doc_id", "")
        chunk_id = r.get("chunk_id", "")
        h = r.get("hash", "")
        items.append(f"{doc_id}:{chunk_id}:{h}")
    s = "|".join(sorted(items))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def generation_params_fingerprint(params: dict) -> str:
    # only include deterministic-relevant params
    keys = sorted(params.keys())
    normalized = {k: params[k] for k in keys}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()


def make_cache_key(model_name: str, model_version: str, system_prompt: str, prompt: str, retrieval_meta, params: dict) -> str:
    parts = [
        f"{model_name}@{model_version}",
        hashlib.sha256((system_prompt or "").encode("utf-8")).hexdigest()[:12],
        hashlib.sha256(canonicalize_prompt(prompt).encode("utf-8")).hexdigest()[:16],
        fingerprint_retrieval(retrieval_meta)[:16],
        generation_params_fingerprint(params)[:16],
    ]
    return "|".join(parts)
