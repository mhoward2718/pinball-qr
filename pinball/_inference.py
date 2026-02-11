"""Backward-compatibility shim — re-exports from ``pinball.linear._inference``."""
from pinball.linear._inference import *  # noqa: F401,F403
from pinball.linear._inference import (  # noqa: F401
    summary,
    InferenceResult,
    _se_iid,
    _se_nid,
    _se_ker,
    _se_rank,
)
