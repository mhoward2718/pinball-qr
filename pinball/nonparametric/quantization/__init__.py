"""Conditional quantile estimation via optimal quantization.

Implements the method of Charlier, Paindaveine & Saracco (2015),
ported from the R ``QuantifQuantile`` package.

References
----------
.. [1] Charlier, I., Paindaveine, D. and Saracco, J. (2015).
       "Conditional quantile estimation through optimal quantization."
       *Journal of Statistical Planning and Inference* 156, 14–30.
"""

from pinball.nonparametric.quantization._clvq import choice_grid

__all__ = ["choice_grid"]
