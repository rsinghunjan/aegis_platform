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
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
"""

    if not provided_key or not _is_api_key_allowed(provided_key, _ALLOWED_CLIENT_KEYS_SET):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

    cid = str(payload.cid).strip()
    if not cid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="cid is required")

    enrolled = _read_enrolled()
    enrolled[cid] = {"cid": cid, "enrolled_at": __import__("time").time(), "revoked": False, "metadata": payload.metadata}
    try:
        _write_enrolled_atomic(enrolled)
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to persist enrollment")

    logger.info("Client enrolled (cid=%s)", cid)
    # Do not echo API keys or other secrets
    return {"status": "enrolled", "cid": cid}


@app.post("/revoke")
async def revoke(
    request: Request,
    payload: dict = Body(...),
    x_api_key: str = Header(None, alias="x-api-key"),
):
    """
    Operator endpoint to revoke an enrolled client:
    Payload: {"cid": "<client-id>", "reason": "optional text"}
    Operator must present x-api-key from ALLOWED_OPERATOR_API_KEYS
    """
    provided_key = x_api_key.strip() if x_api_key else None
    if not _ALLOWED_OPERATOR_KEYS_SET or not provided_key or not _is_api_key_allowed(provided_key, _ALLOWED_OPERATOR_KEYS_SET):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Operator auth required")

    cid = payload.get("cid")
    if not cid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="cid required")
    cid = str(cid).strip()
    enrolled = _read_enrolled()
    rec = enrolled.get(cid)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="client not found")
    rec["revoked"] = True
    rec["revoked_at"] = __import__("time").time()
    rec["revoke_reason"] = payload.get("reason", "")[:1000]
    enrolled[cid] = rec
    try:
        _write_enrolled_atomic(enrolled)
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to persist revoke")
    logger.info("Client revoked (cid=%s)", cid)
    return {"status": "revoked", "cid": cid}


@app.get("/status/{cid}")
async def status(cid: str, x_api_key: str = Header(None, alias="x-api-key")):
    """
    Query enrollment status. For simplicity, require a valid operator key to query status.
    """
    provided_key = x_api_key.strip() if x_api_key else None
    if not _ALLOWED_OPERATOR_KEYS_SET or not provided_key or not _is_api_key_allowed(provided_key, _ALLOWED_OPERATOR_KEYS_SET):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Operator auth required")

    enrolled = _read_enrolled()
    rec = enrolled.get(cid)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="client not enrolled")
    # Return non-sensitive metadata only
    return {"cid": cid, "enrolled_at": rec.get("enrolled_at"), "revoked": bool(rec.get("revoked", False)), "metadata": rec.get("metadata", {})}
aegis_multimodal_ai_system/federated/auth_server.py
