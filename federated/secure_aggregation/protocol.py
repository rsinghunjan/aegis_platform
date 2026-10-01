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
197
198
199
APIs:
    return out


def mask_parameters(param_list: Sequence, my_id: str, peers: Sequence[PeerInfo], privkey: X25519PrivateKey) -> List:
    """
    Mask a list of numpy arrays (param_list) using pairwise masks with peers.

    peers: list of (peer_id, peer_pub_bytes) including self; ordering must be deterministic and identical
           across clients (we sort by peer_id string).
    Behavior:
      For each peer P:
        if P.id == my_id: skip
        if P.id > my_id: add mask_ij
        else: subtract mask_ji
    Returns a new list of arrays in the same shapes containing masked parameters.

    NOTE: All clients must use the exact same ordering of peers for cancellations to work.
    """
    flat, shapes = _flatten_param_list(param_list)
    L = flat.size
    # Sort peers by id deterministically
    peers_sorted = sorted(peers, key=lambda x: str(x[0]))
    my_index = None
    for idx, (pid, _) in enumerate(peers_sorted):
        if str(pid) == str(my_id):
            my_index = idx
            break
    if my_index is None:
        raise RuntimeError("my_id not present in peers listing")

    mask_total = np.zeros_like(flat, dtype="float32")

    # iterate peers and derive shared secrets
    for pid, peer_pub in peers_sorted:
        if str(pid) == str(my_id):
            continue
        # derive shared secret
        try:
            s = derive_shared_secret(privkey, peer_pub)
        except Exception:
            # bad key material; skip (but this will break cancellation)
            raise
        # use deterministic info to ensure both sides derive same mask for pair (i,j)
        info = f"mask::{min(str(my_id), str(pid))}::{max(str(my_id), str(pid))}".encode("utf-8")
        mask_vec = derive_mask_vector(s, L, info=info)
        # Determine sign: + if peer_id > my_id else - (consistent with explained scheme)
        if str(pid) > str(my_id):
            mask_total = mask_total + mask_vec
        else:
            mask_total = mask_total - mask_vec

    masked_flat = flat + mask_total
    masked_list = _unflatten_to_list(masked_flat, shapes)
    return masked_list


# Convenience: serialize / deserialize public keys to/from base64 strings for HTTP transport
def serialize_pubkey(pubbytes: bytes) -> str:
    return base64.urlsafe_b64encode(pubbytes).decode("ascii")


def deserialize_pubkey(b64: str) -> bytes:
    return base64.urlsafe_b64decode(b64.encode("ascii"))
aegis_multimodal_ai_system/federated/secure_aggregation/protocol.py

