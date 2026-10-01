 82
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
"""
        delta = flat_new - flat_old

        # compute L2 norm
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > self.clip_norm and delta_norm > 0:
            delta = delta * (self.clip_norm / delta_norm)

        # add gaussian noise if configured
        if self.noise_multiplier and self.noise_multiplier > 0.0:
            sigma = self.noise_multiplier * self.clip_norm
            noise = np.random.normal(loc=0.0, scale=sigma, size=delta.shape).astype("float32")
            delta = delta + noise

        # reconstruct weights = old + delta
        noisy_flat = flat_old + delta
        new_param_list = _unflatten_to_shapes(noisy_flat, shapes)

        # return updated parameters and number of examples
        return new_param_list, len(self.X), {}

    def evaluate(self, parameters, config):
        _sklearn_set_weights(self.model, parameters)
        preds = self.model.predict(self.X)
        accuracy = float((preds == self.y).mean())
        loss = 1.0 - accuracy
        return float(loss), len(self.X), {"accuracy": accuracy}


def enroll_with_server(enroll_url, cid, api_key, timeout=10):
    logger.info("Enrolling client %s at %s", cid, enroll_url)
    payload = {"cid": cid}
    headers = {"x-api-key": api_key}
    try:
        resp = requests.post(enroll_url, json=payload, headers=headers, timeout=timeout)
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
    parser.add_argument("--cid", type=int, required=True, help="Client id")
    parser.add_argument("--host", type=str, default="localhost:8080", help="Flower server address")
    parser.add_argument("--enroll-host", type=str, default="localhost:8081", help="Enrollment service host (host:port)")
    parser.add_argument("--api-key", type=str, default=os.getenv("CLIENT_API_KEY", ""), help="API key for enrollment")
    parser.add_argument("--clip-norm", type=float, default=float(os.getenv("FL_DP_CLIP_NORM", "1.0")), help="L2 clip norm for client updates")
    parser.add_argument("--noise-multiplier", type=float, default=float(os.getenv("FL_DP_NOISE_MULTIPLIER", "0.0")), help="Noise multiplier for Gaussian noise (0 disables DP)")
    args = parser.parse_args()

    if not args.api_key:
        logger.error("API key not provided. Use --api-key or set CLIENT_API_KEY env var.")
        sys.exit(2)

    enroll_url = f"http://{args.enroll_host}/enroll"
    ok = enroll_with_server(enroll_url, args.cid, args.api_key)
    if not ok:
        logger.error("Enrollment failed; aborting client startup.")
        sys.exit(1)

    # create local toy data
    X, y = make_toy_client_data(n_samples=200, n_features=20, random_state=args.cid)
    client = SklearnNumPyClient(cid=args.cid, X=X, y=y, clip_norm=args.clip_norm, noise_multiplier=args.noise_multiplier)
    fl.client.start_numpy_client(server_address=args.host, client=client)


if __name__ == "__main__":
    main()
