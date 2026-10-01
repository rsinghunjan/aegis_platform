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
"""
Small CLI wrappers for previously top-level scripts.

For each moved top-level script, create a small wrapper that exposes a main()
so you can run it as a module: python -m aegis_multimodal_ai_system.scripts.<name>

Edit the wrappers to call the actual logic in the moved file (or import and call main()).
"""

import importlib
import sys

def _run_module_main(module_name: str, attr: str = "main"):
    """
    Attempt to import module and call `main()` or fall back to running as a script.
    """
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, attr):
            return getattr(mod, attr)()
        # else, if module defines a top-level run() or serve()
        for cand in ("run", "serve"):
            if hasattr(mod, cand):
                return getattr(mod, cand)()
        # Nothing to call
        print(f"Module {module_name} loaded but no {attr}()/run()/serve() found.")
    except Exception as e:
        print(f"Error running module {module_name}: {e}", file=sys.stderr)
        raise

def main_mmaisys_code():
    # module name after moving: aegis_multimodal_ai_system.scripts.mmaisys_code
    return _run_module_main("aegis_multimodal_ai_system.scripts.mmaisys_code")

def main_aegis_storage_code():
    return _run_module_main("aegis_multimodal_ai_system.scripts.aegis_storage_code")

if __name__ == "__main__":
    # basic CLI dispatch: python -m aegis_multimodal_ai_system.scripts.entrypoints <command>
    if len(sys.argv) < 2:
        print("usage: entrypoints <command>")
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "mmaisys_code":
        sys.exit(main_mmaisys_code() or 0)
    elif cmd == "aegis_storage_code":
        sys.exit(main_aegis_storage_code() or 0)
    else:
        print("unknown command", cmd)
        sys.exit(2)
