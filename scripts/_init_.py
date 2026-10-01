  

"""
Package-level script helpers and entrypoints for moved scripts.

After moving top-level scripts into this package, expose module entrypoints like:
  python -m aegis_multimodal_ai_system.scripts.mmaisys_code
  python -m aegis_multimodal_ai_system.scripts.aegis_storage_code

Add wrappers below or import the original script's main() if available.
"""

__all__ = ["mmaisys_code", "aegis_storage_code", "entrypoints"]

from . import entrypoints
