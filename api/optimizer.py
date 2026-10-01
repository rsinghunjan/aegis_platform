"""
Lightweight optimizer helpers used by scripts/optimize_model.py.

Provides:
 - torchscript_to_onnx(torchscript_path, out_path, example_input, opset)
 - quantize_onnx_dynamic(onnx_in, out_dir, method='dynamic')
 - upload_to_s3(paths, bucket, prefix)
 - register_optimized_model(model_name, version, paths, s3_url=None)

These are best-effort helpers that try to call repo registry APIs and return structured results.
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("aegis.api.optimizer")
logging.basicConfig(level=logging.INFO)


def torchscript_to_onnx(torchscript_path: str, out_path: str, example_input, opset: int = 13) -> str:
    """
    Export a TorchScript loaded module to ONNX using torch.onnx.export.
    example_input is a torch.Tensor instance.
    """
    try:
        import torch
    except Exception:
        raise RuntimeError("PyTorch not available for torchscript_to_onnx")
    model = torch.jit.load(torchscript_path, map_location="cpu")
    model.eval()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example_input, out_path, opset_version=opset, input_names=["input"], output_names=["output"])
    logger.info("Exported %s -> %s", torchscript_path, out_path)
    return out_path


def quantize_onnx_dynamic(onnx_in: str, out_dir: str, method: str = "dynamic") -> str:
    """
    Quantize ONNX model using onnxruntime.quantization.quantize_dynamic.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as exc:
        raise RuntimeError("onnxruntime.quantization not available: %s" % exc)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    qname = Path(onnx_in).stem + ".quant.onnx"
    qpath = str(out_dir_p / qname)
    quantize_dynamic(onnx_in, qpath, weight_type=QuantType.QInt8)
    logger.info("Quantized dynamic ONNX %s -> %s", onnx_in, qpath)
    return qpath


def upload_to_s3(paths: List[str], bucket: str, prefix: str, endpoint_url: Optional[str] = None) -> Optional[str]:
    """
    Upload a list of local files to s3://{bucket}/{prefix}/ and return the s3 path of the first file.
    """
    try:
        import boto3
    except Exception:
        raise RuntimeError("boto3 required for upload_to_s3")

    s3 = boto3.client("s3", endpoint_url=endpoint_url,
                      aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
    # ensure bucket exists (best-effort)
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        try:
            s3.create_bucket(Bucket=bucket)
        except Exception:
            logger.exception("Could not create or access bucket %s", bucket)
            raise
    uploaded = []
    for p in paths:
        key = f"{prefix}/{Path(p).name}"
        s3.upload_file(p, bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded[0] if uploaded else None


def register_optimized_model(model_name: str, version: str, paths: List[str], s3_url: Optional[str] = None) -> bool:
    """
    Best-effort registration helper: attempts to call into api.registry to register the optimized artifact.
    """
    try:
        import api.registry as registry
    except Exception:
        logger.warning("api.registry not present; skipping registration")
        return False

    try:
        # common modern API: registry.register_optimized_artifact(model_name, version, paths, s3_url=None)
        if hasattr(registry, "register_optimized_artifact"):
            registry.register_optimized_artifact(model_name, version, paths, s3_url=s3_url)
            logger.info("Called registry.register_optimized_artifact")
            return True
        # fallback to register_artifact
        if hasattr(registry, "register_artifact"):
            registry.register_artifact(model_name, version, paths[0])
            logger.info("Called registry.register_artifact")
            return True
        if hasattr(registry, "register"):
            try:
                registry.register(model_name, version, paths[0])
            except TypeError:
                registry.register(paths[0])
            logger.info("Called registry.register fallback")
            return True
    except Exception:
        logger.exception("Registry registration call failed")
        return False

    return False
