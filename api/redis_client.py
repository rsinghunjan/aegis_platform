"""
Redis client helper with support for standalone, Sentinel and Cluster topologies.

Environment configuration (choose one):
- RATE_LIMIT_REDIS_URL (e.g. redis://localhost:6379/0)
- RATE_LIMIT_REDIS_SENTINELS (comma-separated like host1:26379,host2:26379)
  and RATE_LIMIT_REDIS_SENTINEL_MASTER (name of the master group)
- RATE_LIMIT_REDIS_CLUSTER_NODES (comma-separated like host1:7000,host2:7000)

The helper returns a redis.asyncio.Redis or redis.asyncio.ClusterClient instance.
"""
from typing import Optional
import os
import logging
import urllib.parse

logger = logging.getLogger("aegis.redis_client")

try:
    import redis.asyncio as redis_async
    from redis.asyncio.cluster import RedisCluster
    from redis.asyncio.sentinel import Sentinel
    REDIS_AVAILABLE = True
except Exception:
    redis_async = None
    RedisCluster = None
    Sentinel = None
    REDIS_AVAILABLE = False

_redis_instance = None


async def get_redis() -> Optional[object]:
    """
    Return an async Redis client instance.
    """
    global _redis_instance
    if _redis_instance is not None:
        return _redis_instance

    if not REDIS_AVAILABLE:
        logger.warning("redis.asyncio not installed; Redis integration disabled")
        return None

    # 1) Cluster
    cluster_nodes = os.environ.get("RATE_LIMIT_REDIS_CLUSTER_NODES")
    if cluster_nodes:
        nodes = []
        for n in cluster_nodes.split(","):
            host, _, port = n.strip().partition(":")
            nodes.append({"host": host, "port": int(port or 7000)})
        try:
            rc = RedisCluster(startup_nodes=nodes, decode_responses=True)
            _redis_instance = rc
            logger.info("Connected to Redis Cluster via nodes=%s", nodes)
            return _redis_instance
        except Exception as exc:
            logger.exception("Redis Cluster connection failed: %s", exc)

    # 2) Sentinel
    sentinels = os.environ.get("RATE_LIMIT_REDIS_SENTINELS")
    sentinels_master = os.environ.get("RATE_LIMIT_REDIS_SENTINEL_MASTER")
    if sentinels and sentinels_master:
        try:
            sentinel_hosts = []
            for s in sentinels.split(","):
                host, _, port = s.strip().partition(":")
                sentinel_hosts.append((host, int(port or 26379)))
            sentinel = Sentinel(sentinel_hosts, decode_responses=True)
            master = sentinel.master_for(sentinels_master, decode_responses=True)
            _redis_instance = master
            logger.info("Connected to Redis Sentinel master=%s via %s", sentinels_master, sentinel_hosts)
            return _redis_instance
        except Exception as exc:
            logger.exception("Redis Sentinel connection failed: %s", exc)

    # 3) Single URL
    url = os.environ.get("RATE_LIMIT_REDIS_URL", "redis://localhost:6379/0")
    try:
        # parse URL for TLS or password awareness
        parsed = urllib.parse.urlparse(url)
        # redis.asyncio.from_url is convenient
        client = redis_async.from_url(url, decode_responses=True)
        _redis_instance = client
        logger.info("Connected to Redis via %s", url)
        return _redis_instance
    except Exception as exc:
        logger.exception("Redis connection failed for %s: %s", url, exc)
        return None
