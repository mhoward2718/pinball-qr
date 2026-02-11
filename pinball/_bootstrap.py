"""Backward-compatibility shim — re-exports from ``pinball.linear._bootstrap``."""
from pinball.linear._bootstrap import *  # noqa: F401,F403
from pinball.linear._bootstrap import (  # noqa: F401
    bootstrap,
    BootstrapResult,
    _xy_pairs,
    _wild,
    _mcmb,
)
