import importlib.util

import pytest

HAS_BACKSTITCH = importlib.util.find_spec("pathrag._backstitch") is not None
needs_backstitch = pytest.mark.skipif(
    not HAS_BACKSTITCH, reason="backstitch backend not built (set BACKSTITCH_DIR, run setup.py)")


def kinds(*names):
    """Index kinds as pytest params, Backstitch ones skipped when it isn't built."""
    return [pytest.param(n, marks=needs_backstitch) if n.startswith("backstitch") else n
            for n in names]
