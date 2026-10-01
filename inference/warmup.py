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
"""
Warmup helpers to prime models (load weights into GPU, build kernels, etc).

Usage:
- Call warmup_model(model_wrapper, sample_inputs=..., repeats=3) after model load.
- Keep warmup small but representative of expected input shape to avoid pathologically long warmups.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def warmup_model(model_wrapper, sample_inputs: Optional[List[str]] = None, repeats: int = 3):
    """
    Run a small number of dummy inferences to warm caches.
    model_wrapper is expected to have predict(list[str]) -> list[outputs]
    """
    if model_wrapper is None:
        logger.debug("No model wrapper provided for warmup")
        return
    # Create a small default sample if none provided
    if not sample_inputs:
        sample_inputs = ["warmup"] * 2
    try:
        for i in range(max(1, repeats)):
            logger.debug("Warmup pass %d/%d", i + 1, repeats)
            # model.predict may be sync or async: try sync; callers can adapt
            res = model_wrapper.predict(sample_inputs)
            logger.debug("Warmup pass %d produced %d outputs", i + 1, len(res) if res is not None else 0)
    except Exception:
        logger.exception("Model warmup failed (continuing)")
aegis_multimodal_ai_system/inference/warmup.py
