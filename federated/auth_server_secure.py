 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
"""

@app.post("/enroll")
async def enroll(payload: EnrollRequest = Body(...), x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    # In pilot, we keep simple API-key check; in production require mTLS/OIDC
    # For demo assume ALLOWED_CLIENT_API_KEYS may be set
    allowed_keys = [k.strip() for k in os.getenv("ALLOWED_CLIENT_API_KEYS", "").split(",") if k.strip()]
    provided = x_api_key.strip() if x_api_key else None
    if allowed_keys and provided not in allowed_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    cid = str(payload.cid)
    enrolled = _read_enrolled()
    rec = enrolled.get(cid, {})
    rec["cid"] = cid
    rec["enrolled_at"] = __import__("time").time()
    rec["revoked"] = False
    # Copy metadata but ensure we only store non-sensitive fields; public_key is OK (it's public material)
    md = payload.metadata or {}
    pubkey_b64 = md.get("public_key")
    if pubkey_b64:
        # minimal validation: keep short length
        if len(pubkey_b64) > 2560:
            raise HTTPException(status_code=400, detail="public_key too large")
        rec["public_key"] = pubkey_b64
    rec["metadata"] = {k: v for k, v in md.items() if k != "public_key"}
    enrolled[cid] = rec
    try:
        _write_enrolled_atomic(enrolled)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to persist enrollment")
    return {"status": "enrolled", "cid": cid}


@app.post("/revoke")
async def revoke(payload: dict = Body(...), x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    operator_keys = [k.strip() for k in os.getenv("ALLOWED_OPERATOR_API_KEYS", "").split(",") if k.strip()]
    provided = x_api_key.strip() if x_api_key else None
    if operator_keys and provided not in operator_keys:
        raise HTTPException(status_code=401, detail="Operator auth required")
    cid = str(payload.get("cid", ""))
    if not cid:
        raise HTTPException(status_code=400, detail="cid required")
    enrolled = _read_enrolled()
    rec = enrolled.get(cid)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    rec["revoked"] = True
    rec["revoked_at"] = __import__("time").time()
    enrolled[cid] = rec
    try:
        _write_enrolled_atomic(enrolled)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to persist revoke")
    return {"status": "revoked", "cid": cid}


@app.get("/peers")
async def peers(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    """
    Return a list of enrolled peers and their public keys (base64). For security,
    require operator key or allow clients to request peer list when enrollment keys configured.
    """
    allowed_keys = [k.strip() for k in os.getenv("ALLOWED_CLIENT_API_KEYS", "").split(",") if k.strip()]
    provided = x_api_key.strip() if x_api_key else None
    if allowed_keys and provided not in allowed_keys:
        raise HTTPException(status_code=401, detail="Auth required")
    enrolled = _read_enrolled()
    peers = []
    for cid, rec in enrolled.items():
        if rec.get("revoked"):
            continue
        pub = rec.get("public_key")
        if pub:
            peers.append({"cid": cid, "public_key": pub})
    return {"peers": peers}
aegis_multimodal_ai_system/federated/auth_server_secure.py
