"""Quick smoke tests for the compiled Fortran extension."""
import numpy as np
import pytest

from pinball._native import rqbr, rqfnb

def test_rqbr_smoke():
    """Test rqbr with simple linear data."""
    np.random.seed(42)
    n, p = 20, 2
    X = np.column_stack([np.ones(n), np.linspace(0, 1, n)])
    y = 2.0 + 3.0 * np.linspace(0, 1, n) + np.random.randn(n) * 0.1

    nsol = 2
    ndsol = 2
    params = dict(
        m=n, nn=np.int32(p), m5=np.int32(n + 5), n3=np.int32(p + 3),
        n4=np.int32(p + 4),
        a=np.asfortranarray(X, dtype=np.float64),
        b=np.ascontiguousarray(y, dtype=np.float64),
        t=0.5,
        toler=np.finfo(np.float64).eps ** (2.0 / 3.0),
        ift=np.int32(1),
        x=np.zeros(p, dtype=np.float64),
        e=np.zeros(n, dtype=np.float64),
        s=np.zeros(n, dtype=np.int32),
        wa=np.zeros((n + 5, p + 4), dtype=np.float64),
        wb=np.zeros(n, dtype=np.float64),
        nsol=np.int32(nsol), ndsol=np.int32(ndsol),
        sol=np.zeros((p + 3, nsol), dtype=np.float64),
        dsol=np.zeros((n, ndsol), dtype=np.float64),
        lsol=np.int32(0),
        h=np.zeros((p, nsol), dtype=np.int32),
        qn=np.zeros(p, dtype=np.float64),
        cutoff=np.float64(0),
        ci=np.zeros((4, p), dtype=np.float64),
        tnmat=np.zeros((4, p), dtype=np.float64),
        big=np.finfo(np.float64).max,
        lci1=np.bool_(False),
    )
    raw = rqbr(**params)
    flag = raw[0]
    coef = raw[1]
    print(f"rqbr flag={flag}, coef={coef}")
    # Intercept should be near 2, slope near 3
    assert abs(coef[0] - 2.0) < 0.5, f"Intercept {coef[0]} not near 2.0"
    assert abs(coef[1] - 3.0) < 0.5, f"Slope {coef[1]} not near 3.0"


def test_rqfnb_smoke():
    """Test rqfnb with simple linear data."""
    np.random.seed(42)
    n, p = 20, 2
    X = np.column_stack([np.ones(n), np.linspace(0, 1, n)])
    y = 2.0 + 3.0 * np.linspace(0, 1, n) + np.random.randn(n) * 0.1
    tau = 0.5

    a = np.asfortranarray(X.T, dtype=np.float64)
    c = np.ascontiguousarray(-y, dtype=np.float64)
    rhs = (1.0 - tau) * X.sum(axis=0).astype(np.float64)
    d = np.ones(n, dtype=np.float64)
    u = np.ones(n, dtype=np.float64)
    wn = np.zeros((n, 9), dtype=np.float64, order="F")
    wn[:, 0] = 1.0 - tau
    wp = np.zeros((p, p + 3), dtype=np.float64, order="F")
    nit = np.zeros(3, dtype=np.int32)
    info = np.int32(0)

    (a_out, c_out, rhs_out, d_out, u_out,
     wn_out, wp_out, nit_out, info_out) = rqfnb(
        a, c, rhs, d, u, 0.99995, 1e-6, wn, wp, nit, info,
    )
    print(f"rqfnb info={info_out}, nit={nit_out}, wp[:,0]={wp_out[:, 0]}")
    coef = -wp_out[:, 0]
    print(f"rqfnb coef={coef}")
    # Intercept should be near 2, slope near 3
    assert abs(coef[0] - 2.0) < 0.5, f"Intercept {coef[0]} not near 2.0"
    assert abs(coef[1] - 3.0) < 0.5, f"Slope {coef[1]} not near 3.0"


if __name__ == "__main__":
    test_rqbr_smoke()
    test_rqfnb_smoke()
    print("All smoke tests passed!")
