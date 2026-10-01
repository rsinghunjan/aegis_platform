"""
Model loading/wrapper utilities for Aegis model serving.

Provides BaseModelWrapper (a common load/warmup/predict_batch/cpu_offload
lifecycle) plus TorchModelWrapper and ONNXModelWrapper implementations, and a
small get_preferred_device() helper used by ModelRegistry (api/model_runner.py).
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("aegis.model_loader")

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ort = None
    ORT_AVAILABLE = False


def get_preferred_device() -> str:
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


class BaseModelWrapper:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.device = "cpu"

    def load(self):
        raise NotImplementedError

    def warmup(self, sample_input: Any, iters: int = 1):
        raise NotImplementedError

    def predict_batch(self, batched_inputs: list):
        raise NotImplementedError

    def cpu_offload(self):
        raise NotImplementedError


class TorchModelWrapper(BaseModelWrapper):
    def __init__(self, model_path: str, name: str, version: str, device: Optional[str] = None):
        super().__init__(name, version)
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not available")
        self.model_path = model_path
        self.device = device or get_preferred_device()
        self._model = None
        self._loaded = False

    def load(self):
        logger.info("Loading torch model %s:%s from %s onto %s", self.name, self.version, self.model_path, self.device)
        self._model = torch.load(self.model_path, map_location=self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device)
        self._loaded = True

    def warmup(self, sample_input: Any, iters: int = 1):
        if not self._loaded:
            self.load()
        logger.info("Warming torch model %s:%s", self.name, self.version)
        for _ in range(iters):
            try:
                self.predict_batch([sample_input])
            except Exception:
                logger.exception("torch warmup failed")

    def predict_batch(self, batched_inputs: list):
        if not self._loaded:
            self.load()
        outputs = []
        with torch.no_grad():
            for inp in batched_inputs:
                try:
                    out = self._model(inp) if callable(self._model) else None
                    outputs.append(out)
                except Exception as exc:
                    logger.exception("torch inference error: %s", exc)
                    outputs.append({"error": str(exc)})
        return outputs

    def cpu_offload(self):
        try:
            self._model.cpu()
            self.device = "cpu"
            logger.info("Offloaded model %s:%s to CPU", self.name, self.version)
        except Exception:
            logger.exception("cpu_offload failed")


class ONNXModelWrapper(BaseModelWrapper):
    def __init__(self, model_path: str, name: str, version: str, use_gpu: bool = False):
        super().__init__(name, version)
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not available")
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._sess = None
        self._loaded = False

    def load(self):
        providers = ["CPUExecutionProvider"]
        if self.use_gpu:
            # depending on platform, use CUDAExecutionProvider or others
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Creating ONNXRuntime session %s:%s providers=%s", self.name, self.version, providers)
        self._sess = ort.InferenceSession(self.model_path, providers=providers)
        self._loaded = True

    def warmup(self, sample_input: Dict[str, Any], iters: int = 1):
        if not self._loaded:
            self.load()
        logger.info("Warming ONNX model %s:%s", self.name, self.version)
        for _ in range(iters):
            try:
                in_names = [n.name for n in self._sess.get_inputs()]
                feed = {}
                # naive mapping: sample_input must be dict of input_name -> numpy array
                for nm in in_names:
                    if nm in sample_input:
                        feed[nm] = sample_input[nm]
                self._sess.run(None, feed)
            except Exception:
                logger.exception("onnx warmup failed")

    def predict_batch(self, batched_inputs: list):
        if not self._loaded:
            self.load()
        outputs = []
        in_names = [n.name for n in self._sess.get_inputs()]
        # batched_inputs: list of dicts keyed by input names
        for inp in batched_inputs:
            try:
                feed = {k: v for k, v in inp.items() if k in in_names}
                out = self._sess.run(None, feed)
                outputs.append(out)
            except Exception as exc:
                logger.exception("onnx inference error: %s", exc)
                outputs.append({"error": str(exc)})
        return outputs

    def cpu_offload(self):
        # ONNXRuntime sessions don't hold a distinct GPU/CPU device handle we
        # can migrate the way torch tensors do; dropping the session forces a
        # reload (and re-warmup) on next use.
        self._sess = None
        self._loaded = False
        logger.info("Released ONNX session for %s:%s", self.name, self.version)
