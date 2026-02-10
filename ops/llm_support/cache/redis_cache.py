"""
Redis-backed cache wrapper.

Key design:
- Stores JSON-serializable dict: {response: str, meta: {...}}
- TTL controlled when setting keys
- Simple prefixing by namespace to support invalidation by model_version or policy_version
"""
import json
import logging
from typing import Optional
import redis

LOG = logging.getLogger("aegis.cache.redis")


class RedisCache:
    def __init__(self, url: str = "redis://localhost:6379/0", namespace: str = "aegis:cache"):
        self._r = redis.from_url(url, decode_responses=True)
        self._ns = namespace

    def _pref(self, key: str) -> str:
        return f"{self._ns}:{key}"

    def get(self, key: str) -> Optional[dict]:
        try:
            val = self._r.get(self._pref(key))
            if not val:
                return None
            return json.loads(val)
        except Exception:
            LOG.exception("redis get failed")
            return None

    def set(self, key: str, value: dict, ttl: int = None):
        try:
            v = json.dumps(value)
            if ttl:
                self._r.setex(self._pref(key), ttl, v)
            else:
                self._r.set(self._pref(key), v)
        except Exception:
            LOG.exception("redis set failed")

    def delete(self, key: str):
        try:
            self._r.delete(self._pref(key))
        except Exception:
            LOG.exception("redis delete failed")

    def invalidate_namespace_prefix(self, prefix: str):
        """
        Invalidate keys that start with prefix.
        Warning: uses SCAN; in production use key design that allows tag/index invalidation.
        """
        try:
            pattern = f"{self._ns}:{prefix}*"
            cur = "0"
            while cur != 0 and cur != "0":
                cur, keys = self._r.scan(cur, match=pattern, count=1000)
                if keys:
                    self._r.delete(*keys)
        except Exception:
            LOG.exception("redis invalidate failed")
