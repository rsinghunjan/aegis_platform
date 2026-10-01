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
Model eviction & memory manager for Aegis.
                if ":" in wdev:
                    try:
                        widx = int(wdev.split(":")[1])
                    except Exception:
                        widx = 0
                else:
                    widx = 0
                if widx != dev:
                    continue
            else:
                continue
            last_used = getattr(wrapper, "_last_used", None) or 0
            # approximate memory footprint if wrapper exposes `approx_memory_mb`
            approx_mb = getattr(wrapper, "approx_memory_mb", None) or 0
            candidates.append((k, last_used, approx_mb, wrapper))

        if not candidates:
            logger.info("No eviction candidates on device %s", dev)
            return

        # choose strategy
        if STRATEGY == "largest":
            candidates.sort(key=lambda t: t[2], reverse=True)
        elif STRATEGY == "oldest":
            candidates.sort(key=lambda t: t[1])
        else:
            # default LRU: sort by last_used ascending (oldest first)
            candidates.sort(key=lambda t: t[1])

        freed = 0
        target_free_mb = int(stats["total"] * SOFT_WM)  # target used fraction -> compute target used; we want used <= soft watermark
        target_free_mb = stats["total"] - int(stats["total"] * SOFT_WM)
        # attempt evictions until free >= MIN_FREE_MB or used <= soft watermark
        for k, last_used, approx_mb, wrapper in candidates:
            if stats["free"] + freed >= max(MIN_FREE_MB, target_free_mb):
                break
            try:
                # prefer cheap offload
                if hasattr(wrapper, "cpu_offload"):
                    logger.info("Offloading model %s (approx_mb=%s) from GPU %s", k, approx_mb, dev)
                    try:
                        wrapper.cpu_offload()
                        freed += approx_mb or 0
                        if METRICS_AVAILABLE:
                            EVICTION_COUNT.inc()
                            EVICTION_ACTIONS.labels(action="cpu_offload").inc()
                    except Exception:
                        logger.exception("cpu_offload failed for %s; trying unload", k)
                        # fallthrough to unload
                # final fallback: unload from registry
                logger.info("Unloading model %s from registry", k)
                try:
                    # registry.unload handles stopping batchers etc.
                    self.registry.unload(*k.split(":", 1))
                    freed += approx_mb or 0
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
