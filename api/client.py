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
 60
 61
 62
 63
 64
"""
Tiny helper client for Aegis inference API (dev convenience).

This is not a full generated SDK — it's a compact wrapper showing how to call
versioned endpoints, attach Bearer token and handle JSON vs multipart.
"""
import base64
import requests
from typing import Optional, Dict, Any

class AegisClient:
    def __init__(self, base_url: str = "http://localhost:8080", token: Optional[str] = None, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout

    def _headers(self, extra: Optional[Dict[str,str]] = None):
        h = {"Accept": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        if extra:
            h.update(extra)
        return h

    def predict(self, model: str, version: str, text: Optional[str] = None,
                image_bytes: Optional[bytes] = None, audio_bytes: Optional[bytes] = None,
                parameters: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        url = f"{self.base}/v1/models/{model}/versions/{version}/predict"
        # prefer JSON: encode files as base64 for the example (not ideal for large files)
        payload = {}
        if text:
            payload["text"] = text
        if parameters:
            payload["parameters"] = parameters
        if image_bytes:
            payload["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
        if audio_bytes:
            payload["audio_base64"] = base64.b64encode(audio_bytes).decode("utf-8")
        headers = self._headers({"Content-Type": "application/json"})
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def predict_multipart(self, model: str, version: str, text: Optional[str] = None,
                          image_path: Optional[str] = None, audio_path: Optional[str] = None) -> Dict[str,Any]:
        url = f"{self.base}/v1/models/{model}/versions/{version}/predict-multipart"
        files = {}
        data = {}
        if text:
            data["text"] = text
        if image_path:
            files["image_file"] = open(image_path, "rb")
        if audio_path:
            files["audio_file"] = open(audio_path, "rb")
        headers = self._headers()
        r = requests.post(url, data=data, files=files, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        # close files
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass
        return r.json()
api/client.py
