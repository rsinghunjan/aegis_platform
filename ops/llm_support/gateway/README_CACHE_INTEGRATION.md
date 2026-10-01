# Streaming proxy cache integration notes

- streaming_proxy_cached.py demonstrates a conservative integration with local/redis/hybrid modes.
- It streams backend chunks incrementally to clients for low-latency UX and accumulates the final response before deciding to cache.
- For heavy production usage, consider:
  - Using a Redis cluster with tag/index based invalidation
  - Adding provenance metadata & signing when caching (audit)
  - Ensuring DP charging semantics align with cache policy (charge on hit or disallow caching)
  - Limiting maximum cached response size to avoid memory/redis abuse
