"""Container entrypoint for invoking a tool shipped in its pinned OCI image."""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json
import os
import re
import resource
import shutil
import sys


def main() -> None:
    if len(sys.argv) != 2 or ":" not in sys.argv[1]:
        raise SystemExit("expected module:function")
    module_name, function_name = sys.argv[1].split(":", 1)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module_name) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", function_name
    ):
        raise SystemExit("invalid module:function")

    request = json.load(sys.stdin)
    if not isinstance(request, dict) or not isinstance(
        request.get("environment"), dict
    ):
        raise SystemExit("invalid sandbox request")
    for name, value in request["environment"].items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or not isinstance(value, str)
        ):
            raise SystemExit("invalid sandbox environment")
        os.environ[name] = value

    with contextlib.redirect_stdout(sys.stderr):
        function = getattr(importlib.import_module(module_name), function_name, None)
        if not callable(function):
            raise SystemExit("sandbox entrypoint is not callable")
        result = function(request.get("payload"))
        if inspect.isawaitable(result):
            result = asyncio.run(result)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    response = {
        "result": result,
        "resource_usage": {
            "cpu_seconds": usage.ru_utime + usage.ru_stime,
            "memory_peak_bytes": usage.ru_maxrss * 1024,
            "disk_used_bytes": shutil.disk_usage("/tmp").used,
        },
    }
    sys.stdout.write(json.dumps(response, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
