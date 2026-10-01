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
import logging
import json
import sys
from typing import Optional, Dict, Any

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, Any] = {
            "ts": getattr(record, "ts", None) or self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        # include extra fields (request_id, trace_id, model_version etc.) if provided
        for k, v in getattr(record, "__dict__", {}).items():
            if k not in ("msg","args","levelname","levelno","name","module","funcName","lineno","ts"):
                # keep only simple serializable types
                try:
                    json.dumps({k:v})
                    data[k] = v
                except Exception:
                    data[k] = str(v)
        return json.dumps(data, ensure_ascii=False)

def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    name = name or __name__
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
