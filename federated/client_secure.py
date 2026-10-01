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
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
"""
                try:
                    pk = deserialize_pubkey(pk_b64)
                except Exception:
                    continue
                peers.append((pcid, pk))
            # include self in peers list to ensure consistent ordering across clients
            peers.append((self.cid, deserialize_pubkey(self.pub_b64)))
            # sort and store
            peers_sorted = sorted(peers, key=lambda x: str(x[0]))
            self.peers = peers_sorted
            logger.info("Retrieved %d peers for secure aggregation", len(self.peers))
        except Exception as e:
            logger.exception("Error fetching peers: %s", e)


def enroll_with_server(enroll_url, cid, api_key, pub_b64, timeout=10):
    logger.info("Enrolling client %s at %s (publishing public_key)", cid, enroll_url)
    payload = {"cid": cid, "metadata": {"public_key": pub_b64}}
    headers = {"x-api-key": api_key} if api_key else {}
    try:
        resp = __import__("requests").post(enroll_url, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        logger.error("Enrollment request failed: %s", e)
        return False
    if resp.status_code != 200:
        logger.error("Enrollment failed: %s %s", resp.status_code, resp.text)
        return False
    logger.info("Enrollment successful for client %s", cid)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--host", type=str, default="localhost:8080", help="Flower server address")
    parser.add_argument("--enroll-host", type=str, default="localhost:8081", help="Enrollment service host (host:port)")
    parser.add_argument("--api-key", type=str, default=os.getenv("CLIENT_API_KEY", ""), help="API key for enrollment")
    parser.add_argument("--clip-norm", type=float, default=float(os.getenv("FL_DP_CLIP_NORM", "1.0")), help="L2 clip norm")
    parser.add_argument("--noise-multiplier", type=float, default=float(os.getenv("FL_DP_NOISE_MULTIPLIER", "0.0")), help="Noise multiplier (0 disables DP)")
    args = parser.parse_args()

    # generate keypair
    try:
        priv, pub = generate_keypair()
    except Exception as e:
        logger.error("Failed to generate keypair: %s", e)
        sys.exit(2)
    pub_b64 = serialize_pubkey(pub)

    enroll_url = f"http://{args.enroll_host}"
    ok = enroll_with_server(f"{enroll_url}/enroll", args.cid, args.api_key, pub_b64)
    if not ok:
        logger.error("Enrollment failed; aborting client startup.")
        sys.exit(1)

    X, y = make_toy_client_data(n_samples=200, n_features=20, random_state=args.cid)
    client = SecureSklearnClient(cid=args.cid, X=X, y=y, privkey=priv, pub_b64=pub_b64, enroll_url=enroll_url, clip_norm=args.clip_norm, noise_multiplier=args.noise_multiplier)
    fl.client.start_numpy_client(server_address=args.host, client=client)


if __name__ == "__main__":
    main()
aegis_multimodal_ai_system/federated/client_secure.py
