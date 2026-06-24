import importlib.util
import os


if os.environ.get("VERL_DISABLE_BROKEN_DEEP_EP") == "1":
    _orig_find_spec = importlib.util.find_spec

    def _find_spec_without_deep_ep(name, package=None):
        if name == "deep_ep" or name.startswith("deep_ep."):
            return None
        return _orig_find_spec(name, package)

    importlib.util.find_spec = _find_spec_without_deep_ep
