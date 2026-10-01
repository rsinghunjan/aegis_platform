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
"""
    The aggregate_fit override expects results as List[Tuple[ClientProxy, flwr.server.client_manager.FitRes]]
    (Flower internals may vary by version; this implementation uses client_proxy.client_id or client_proxy.cid)
    """
    def aggregate_fit(self, rnd: int, results, failures):
        """
        Filter out results coming from clients not currently enrolled or revoked.
        Then call parent aggregate_fit with the allowed results.
        """
        enrolled = _load_enrolled()
        allowed = []
        if results is None:
            return None

        filtered_count = 0
        # results is a list of tuples (client_proxy, fit_res) in newer Flower versions
        for item in results:
            try:
                # Unpack; Flower may provide (client_proxy, fit_res) or (cid, params, num_examples)
                if isinstance(item, tuple) and len(item) == 2 and hasattr(item[0], "cid"):
                    client_proxy: ClientProxy = item[0]
                    fit_res = item[1]
                    cid = getattr(client_proxy, "cid", None) or getattr(client_proxy, "client_id", None) or getattr(client_proxy, "id", None)
                else:
                    # Unknown tuple shape; keep conservative and include
                    allowed.append(item)
                    continue
            except Exception:
                allowed.append(item)
                continue

            # Check enrollment record
            rec = enrolled.get(str(cid))
            if not rec:
                logger.warning("Aggregating skipping client %s: not enrolled", cid)
                filtered_count += 1
                continue
            if rec.get("revoked"):
                logger.warning("Aggregating skipping client %s: revoked", cid)
                filtered_count += 1
                continue
            # allowed
            allowed.append(item)

        if filtered_count:
            logger.info("Filtered out %d client updates due to enrollment checks", filtered_count)

        # Call base aggregate on allowed results
        return super().aggregate_fit(rnd, allowed, failures)


def main():
    # Start metrics server
    start_metrics_server(port=8000)
    logger.info("Prometheus metrics server started on :8000")

    # Start enrollment service
    _start_enroll_service(host="0.0.0.0", port=8081)

    strategy = EnrollmentFilteredFedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=1,
        min_eval_clients=1,
        min_available_clients=1,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
aegis_multimodal_ai_system/federated/server.py
