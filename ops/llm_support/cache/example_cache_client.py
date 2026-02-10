"""
Example usage:
    python3 example_cache_client.py
"""
from ops.llm_support.cache.key_utils import make_cache_key
from ops.llm_support.cache.cache import LocalCache

def main():
    cache = LocalCache(maxsize=100, ttl=60)
    key = make_cache_key("test-model","v1","","hello","",{"temperature":0.0})
    cache.set(key, {"response":"hello world"})
    v = cache.get(key)
    print("cached:", v)

if __name__ == "__main__":
    main()
