"""
Unit tests for carbon scheduler: local override and TTL cache behavior.
"""

import os
import time

import pytest

pytest.importorskip("aegis_multimodal_ai_system.carbon.carbon_scheduler")

from aegis_multimodal_ai_system.carbon.carbon_scheduler import (
    get_current_intensity,
    should_schedule_now,
    _CACHE,
    _CACHE_LOCK,
)


def test_local_override():
    os.environ["CARBON_LOCAL_INTENSITY"] = "42.5"
    # reset cache
    with _CACHE_LOCK:
        _CACHE.update({"ts": 0, "value": None})
    val = get_current_intensity()
    assert val == 42.5
    assert should_schedule_now(50) is True
    del os.environ["CARBON_LOCAL_INTENSITY"]


def test_cache_ttl_behavior():
    # set a value in cache and ensure it's returned until TTL expires
    with _CACHE_LOCK:
        _CACHE.update({"ts": int(time.time()), "value": 10.0})
    val = get_current_intensity()
    assert val == 10.0
