  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
"""
Optional backend that writes audit events directly to OpenSearch/Elasticsearch.

Usage: configure AUDIT_BACKEND=elasticsearch and set AUDIT_ES_URL and AUDIT_ES_INDEX.
This is useful for low-latency search. For long-term retention, still write to S3 and index asynchronously.

Note: this uses the official elasticsearch client; if you use OpenSearch, use opensearch-py and adjust auth.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)
try:
    from elasticsearch import Elasticsearch, helpers  # type: ignore
except Exception:
    Elasticsearch = None  # type: ignore


ES_URL = os.getenv("AUDIT_ES_URL", "")
ES_INDEX = os.getenv("AUDIT_ES_INDEX", "aegis-audit")

class ElasticsearchBackend:
    def __init__(self, url: str = None, index: str = None):
        if Elasticsearch is None:
            raise RuntimeError("elasticsearch client not installed; pip install elasticsearch or opensearch-py")
        self.url = url or ES_URL
        self.index = index or ES_INDEX
        self.client = Elasticsearch([self.url])

        # Create index with mapping if missing
        if not self.client.indices.exists(index=self.index):
            mapping = {
                "mappings": {
                    "properties": {
                        "request_id": {"type": "keyword"},
                        "timestamp": {"type": "date", "format": "epoch_second"},
                        "model_version": {"type": "keyword"},
                        "flagged": {"type": "boolean"},
                        "reason": {"type": "keyword"},
                        "text_snippet": {"type": "text", "index": True},
                    }
                }
            }
            self.client.indices.create(index=self.index, body=mapping)

    def write(self, event: Dict[str, Any]) -> None:
        try:
            eid = event.get("request_id") or str(int(time.time()*1000))
            # convert timestamp to seconds (ES expects seconds for epoch_second)
            if "timestamp" not in event:
                event["timestamp"] = time.time()
            self.client.index(index=self.index, id=eid, document=event)
        except Exception:
            logger.exception("ElasticsearchBackend: failed to index audit event")
aegis_multimodal_ai_system/audit/elasticsearch_backend.py
