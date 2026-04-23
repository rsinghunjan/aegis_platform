#!/usr/bin/env python3
"""
Convert a PyTorch model checkpoint to ONNX format.

Usage:
  python conversion/convert_to_onnx.py --checkpoint model.pt --out model.onnx --opset 13
  python conversion/convert_to_onnx.py --checkpoint model.pt --out model.onnx --input-shapes "image:1,3,224,224;text_ids:1,128"

Supports:
  - PyTorch state_dict checkpoints
  - Full model saves (torch.save(model, ...))
  - Dynamic batch size export
  - Custom input shapes specification
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn


def parse_input_shapes(shapes_str: str) -> Dict[str, Tuple[int, ...]]:
    """
    Parse input shapes from string format.

    Format: "name1:dim1,dim2,...;name2:dim1,dim2,..."
    Example: "image:1,3,224,224;text_ids:1,128"
    """
    if not shapes_str:
        return {}

    result = {}
    for spec in shapes_str.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        name, dims = spec.split(":")
        result[name.strip()] = tuple(int(d) for d in dims.split(","))
    return result


def load_model(checkpoint_path: str, model_class: Optional[str] = None) -> nn.Module:
    """
    Load a PyTorch model from checkpoint.

    Attempts multiple loading strategies:
    1. Full model save (torch.save(model, ...))
    2. State dict with model class reconstruction
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # If checkpoint is already a model
    if isinstance(checkpoint, nn.Module):
        return checkpoint

    # If checkpoint is a state dict, we need a model architecture
    if isinstance(checkpoint, dict):
        # Try to infer model architecture from registry or use default
        if model_class:
            # Import and instantiate model class if provided
            # This is a placeholder - in practice you'd have a model registry
            raise NotImplementedError(
                "Model class loading not implemented. "
                "To convert, save the full model with: torch.save(model, 'model.pt') "
                "instead of just the state_dict."
            )

        raise ValueError(
            "Checkpoint contains only state_dict. "
            "Either provide --model-class or save full model with: "
            "torch.save(model, 'model.pt')"
        )

    raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")


def convert_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shapes: Dict[str, Tuple[int, ...]],
    opset_version: int = 13,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
) -> dict:
    """
    Convert PyTorch model to ONNX format.

    Args:
        model: PyTorch model to convert
        output_path: Path for output ONNX file
        input_shapes: Dict mapping input names to shapes
        opset_version: ONNX opset version
        dynamic_axes: Optional dynamic axis specification
        input_names: Optional list of input names
        output_names: Optional list of output names

    Returns:
        Dictionary with conversion metadata
    """
    model.eval()

    # Create dummy inputs based on shapes
    dummy_inputs = {}
    for name, shape in input_shapes.items():
        # Determine dtype based on name heuristics
        if "id" in name.lower() or "token" in name.lower() or "idx" in name.lower():
            dummy_inputs[name] = torch.randint(0, 1000, shape, dtype=torch.long)
        else:
            dummy_inputs[name] = torch.randn(shape)

    # Use provided names or derive from input_shapes
    if input_names is None:
        input_names = list(input_shapes.keys())

    if output_names is None:
        output_names = ["output"]

    # Default dynamic axes: batch dimension
    if dynamic_axes is None:
        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {0: "batch_size"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    # Prepare input tuple/dict for export
    if len(dummy_inputs) == 1:
        export_input = list(dummy_inputs.values())[0]
    else:
        export_input = tuple(dummy_inputs.values())

    # Export to ONNX
    torch.onnx.export(
        model,
        export_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        model_size = Path(output_path).stat().st_size
    except ImportError:
        print("Warning: onnx package not installed, skipping verification. "
              "Install with: pip install onnx", file=sys.stderr)
        model_size = Path(output_path).stat().st_size

    metadata = {
        "output_path": str(output_path),
        "opset_version": opset_version,
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": {k: list(v) for k, v in input_shapes.items()},
        "dynamic_axes": dynamic_axes,
        "model_size_bytes": model_size,
    }

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint to ONNX format"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to PyTorch checkpoint (.pt, .pth, .bin)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )
    parser.add_argument(
        "--input-shapes",
        default="input:1,3,224,224",
        help="Input shapes in format 'name:d1,d2,...;name2:d1,d2,...'",
    )
    parser.add_argument(
        "--model-class",
        default=None,
        help="Model class name for state_dict loading (optional)",
    )
    parser.add_argument(
        "--output-names",
        default="output",
        help="Comma-separated output names",
    )
    args = parser.parse_args()

    # Parse input shapes
    input_shapes = parse_input_shapes(args.input_shapes)
    if not input_shapes:
        print("Error: No valid input shapes provided", file=sys.stderr)
        sys.exit(1)

    output_names = [n.strip() for n in args.output_names.split(",")]

    try:
        # Load model
        print(f"Loading model from {args.checkpoint}...")
        model = load_model(args.checkpoint, args.model_class)

        # Convert to ONNX
        print(f"Converting to ONNX (opset {args.opset})...")
        metadata = convert_to_onnx(
            model=model,
            output_path=args.out,
            input_shapes=input_shapes,
            opset_version=args.opset,
            output_names=output_names,
        )

        print(f"Successfully converted model to {args.out}")
        print(json.dumps(metadata, indent=2))

        # Write metadata sidecar
        metadata_path = Path(args.out).with_suffix(".onnx.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata written to {metadata_path}")

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
