"""`stepy` forms ada = A diag(d) A' and Choleski-solves ada x = b.

It is the dominant cost of a Frisch-Newton iteration (25-32%, measured), and it
accumulates with BLAS-3 `dsyrk` over cache-sized column blocks rather than n
rank-1 `dsyr` updates.  These tests pin the two things that can go wrong with a
blocked rewrite: the block boundaries, and the `d < 0` fallback.

Note `dposv` overwrites `ada` with its upper Cholesky factor -- not the Gram
matrix -- so that is what the reference compares against.
"""

import numpy as np
import pytest

try:
    from scipy.linalg import cho_solve, cholesky
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def _has_native():
    try:
        from pinball._native import stepy  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _has_native(), reason="Fortran extension not built"),
    pytest.mark.skipif(not _HAS_SCIPY, reason="scipy required"),
]

# Mirrors `nb = 16384/p` in fortran/rqfnb.f: the ~128 KiB block width.
def _nb(p):
    return max(1, 16384 // p)


def _run(p, n, seed):
    from pinball._native import stepy

    rng = np.random.default_rng(seed)
    a = np.asfortranarray(rng.standard_normal((p, n)))
    d = np.asfortranarray(rng.random(n) + 0.05)
    b = np.asfortranarray(rng.standard_normal(p))
    ada = np.zeros((p, p), order="F")
    x = b.copy()
    stepy(a, d, x, ada, np.int32(0))
    return a, d, b, ada, x


@pytest.mark.parametrize("p", [1, 2, 5, 20, 64])
@pytest.mark.parametrize("offset", [-1, 0, 1, 7])
def test_matches_reference_across_block_boundaries(p, offset):
    """The blocking is the risk, so test n either side of a block edge and at a
    multiple of it -- an off-by-one in the tail would show up nowhere else."""
    nb = _nb(p)
    n = nb + offset if offset <= 1 else 3 * nb + offset
    if n < p:
        pytest.skip("need n >= p")
    a, d, b, ada, x = _run(p, n, seed=p * 1000 + n % 977)

    gram = (a * d) @ a.T
    R = cholesky(gram, lower=False)
    np.testing.assert_allclose(np.triu(ada), np.triu(R), atol=1e-12)
    np.testing.assert_allclose(x, cho_solve((R, False), b), atol=1e-10)


def test_single_column():
    """n == 1: the smallest possible block."""
    a, d, b, ada, x = _run(p=1, n=1, seed=0)
    gram = (a * d) @ a.T
    np.testing.assert_allclose(ada[0, 0], np.sqrt(gram[0, 0]), atol=1e-12)


def test_negative_weights_take_the_rank_one_fallback():
    """dsyrk needs sqrt(d).  lpfnb always passes d > 0, but the routine must
    stay correct for any caller, so negative weights fall back to dsyr."""
    from pinball._native import stepy

    p, n = 4, 500
    rng = np.random.default_rng(3)
    a = np.asfortranarray(rng.standard_normal((p, n)))
    d = np.asfortranarray(rng.random(n) + 0.5)
    d[17] = -0.25                                    # forces the fallback
    # Keep the Gram matrix positive definite so dposv still succeeds.
    b = np.asfortranarray(rng.standard_normal(p))
    ada = np.zeros((p, p), order="F")
    x = b.copy()
    stepy(a, d, x, ada, np.int32(0))

    gram = (a * d) @ a.T
    R = cholesky(gram, lower=False)
    np.testing.assert_allclose(np.triu(ada), np.triu(R), atol=1e-10)
    np.testing.assert_allclose(x, cho_solve((R, False), b), atol=1e-8)
