"""OpenEnv validation shim exposing the FastAPI app at server/app.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT_SERVER = Path(__file__).resolve().parents[1] / "server.py"
_SPEC = importlib.util.spec_from_file_location("creditmaze_root_server", _ROOT_SERVER)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

app = _MODULE.app


def main():
    return _MODULE.main()


if __name__ == "__main__":
    main()
