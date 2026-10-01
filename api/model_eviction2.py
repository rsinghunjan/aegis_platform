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
 - inspects GPU memory (torch.cuda) and evicts least-recently-used models when high watermark reached
            free = total - used
            return {"total": total, "used": used, "free": free}

    async def _attempt_eviction(self, dev: int, torch, stats):
        candidates = []
        for k, wrapper in list(getattr(self.registry, "_models", {}).items()):
            wdev = getattr(wrapper, "device", None)
            if not wdev:
                continue
            # match device index if wrapper exposes "cuda:idx"
            if isinstance(wdev, str) and wdev.startswith("cuda"):
                try:
                    if ":" in wdev:
                        widx = int(wdev.split(":")[1])
                    else:
                        widx = 0
                except Exception:
                    widx = 0
                if widx != dev:
                    continue
            else:
                continue
            last_used = getattr(wrapper, "_last_used", 0)
            approx_mb = getattr(wrapper, "approx_memory_mb", 0) or 0
            candidates.append((k, last_used, approx_mb, wrapper))

        if not candidates:
            logger.info("No eviction candidates on device %s", dev)
            return

        if STRATEGY == "largest":
            candidates.sort(key=lambda t: t[2], reverse=True)
        elif STRATEGY == "oldest":
            candidates.sort(key=lambda t: t[1])
        else:
            candidates.sort(key=lambda t: t[1])  # LRU by last_used ascending

        freed = 0
        target_free_mb = int(stats["total"] - int(stats["total"] * SOFT_WM))
        for k, last_used, approx_mb, wrapper in candidates:
            if stats["free"] + freed >= max(MIN_FREE_MB, target_free_mb):
                break
            try:
                if hasattr(wrapper, "cpu_offload"):
                    logger.info("Offloading model %s (approx_mb=%s) from GPU %s", k, approx_mb, dev)
                    try:
                        wrapper.cpu_offload()
                        freed += approx_mb
                        if METRICS_AVAILABLE:
                            EVICTION_COUNT.inc()
                            EVICTION_ACTIONS.labels(action="cpu_offload").inc()
                        continue
                    except Exception:
                        logger.exception("cpu_offload failed for %s; trying unload", k)
                # fallback to unload from registry
                name, version = k.split(":", 1)
                logger.info("Unloading model %s from registry", k)
                try:
                    self.registry.unload(name, version)
                    freed += approx_mb
                    if METRICS_AVAILABLE:
                        EVICTION_COUNT.inc()
                        EVICTION_ACTIONS.labels(action="unload").inc()
                except Exception:
                    logger.exception("registry.unload failed for %s", k)
            except Exception:
                logger.exception("eviction candidate handling failed for %s", k)

        if freed:
            logger.info("Eviction freed approx %s MB on device %s", freed, dev)
        else:
            logger.warning("Eviction attempted but freed no measurable memory on device %s", dev)
api/model_eviction.py
