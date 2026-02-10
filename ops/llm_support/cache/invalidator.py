# Aegis Prompt/Response Caching

This folder implements three caching modes for prompt/response caching:
- Local in-memory LRU + TTL cache (fast, per-pod)
- Redis-backed shared cache (scalable, cross-pod)
- Hybrid mode: local first, fallback to Redis, write-through on miss

Configuration (env vars)
- CACHE_MODE: "local" | "redis" | "hybrid" (default: local)
- LOCAL_CACHE_SIZE: entries for local cache
- LOCAL_CACHE_TTL: seconds TTL for cached entries
- REDIS_URL: redis://...
- CACHE_SAFE_SCORE: maximum safety score allowed to cache (default 0.1)
- CACHE_MAX_TEMPERATURE: maximum temperature allowed to cache (default 0.0)
- CACHE_DP_ALLOW_HITS: if true, cached hits for DP-charged requests will still attempt DP charge
- METRICS_PORT: prometheus metrics server port

Important safety rules (defaults are conservative)
- DP-charged responses are NOT cached unless CACHE_DP_ALLOW_HITS=true
- Only deterministic/low-temperature requests are cached
- Only responses with safety_score <= CACHE_SAFE_SCORE are cached
- Model or policy changes should call the invalidation webhook `/invalidate` to evict stale cached entries

Integration
- streaming_proxy_cached.py provides an example integration into the streaming proxy
- invalidator.py provides an HTTP endpoint to evict keys when a model/policy changes
- key_utils.py provides canonical key generation utilities

Metrics
- aegis_cache_hits_total{backend="local|redis|backend"}
- aegis_cache_misses_total{backend="local|redis|backend"}
- aegis_cache_latency_seconds
- aegis_local_cache_size
- aegis_cache_redis_up

Next steps / production hardening
- Use a Redis cluster and implement tag-based invalidation (index tags) for scalable invalidation
- Add provenance logging into audit store when writing cache entries (cosign sign evidence bundle)
- Add stricter TTL and size guards for large responses; enforce maximum cached response size
