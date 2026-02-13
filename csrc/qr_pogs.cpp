/*
 * qr_pogs.cpp — Quantile regression adapter for POGS.
 *
 * Maps the pinball loss to POGS graph-form and calls PogsD.
 * This is intentionally minimal — ~60 lines of logic.
 *
 * Graph-form:
 *   minimise  sum_i f_i(y_i)  +  sum_j g_j(x_j)
 *   s.t.      y = A x
 *
 * where:
 *   f_i(y_i) = 0.5 * |y_i - b_i| + (0.5 - tau) * y_i
 *            = c * h(a * y_i - b_i) + d * y_i + e * y_i^2
 *     with  h = kAbs, a = 1, b = y[i], c = 0.5, d = 0.5-tau, e = 0
 *
 *   g_j(x_j) = 0
 *     with  h = kZero, a = 1, b = 0, c = 1, d = 0, e = 0
 */

#include "qr_pogs.h"
#include "pogs/interface_c/pogs_c.h"

#include <cstdlib>
#include <vector>

extern "C"
int qr_pogs_solve(
    size_t m, size_t n,
    const double *A, const double *y,
    double tau,
    double abs_tol, double rel_tol,
    unsigned int max_iter,
    unsigned int verbose,
    int adaptive_rho,
    double rho,
    double *beta,
    double *optval,
    unsigned int *final_iter)
{
    /* ── f functions (row objectives): one per observation ── */
    std::vector<double> f_a(m, 1.0);
    std::vector<double> f_b(y, y + m);         /* b_i = y[i]         */
    std::vector<double> f_c(m, 0.5);           /* c = 0.5            */
    std::vector<double> f_d(m, 0.5 - tau);     /* d = 0.5 - tau      */
    std::vector<double> f_e(m, 0.0);
    std::vector<enum FUNCTION> f_h(m, ABS);    /* kAbs = ABS in C API */

    /* ── g functions (column objectives): one per predictor ── */
    std::vector<double> g_a(n, 1.0);
    std::vector<double> g_b(n, 0.0);
    std::vector<double> g_c(n, 1.0);
    std::vector<double> g_d(n, 0.0);
    std::vector<double> g_e(n, 0.0);
    std::vector<enum FUNCTION> g_h(n, ZERO);   /* kZero = ZERO */

    /* ── Output work arrays ── */
    std::vector<double> y_out(m);
    std::vector<double> l_out(m);

    /* ── Call POGS ── */
    int status = PogsD(
        ROW_MAJ, m, n, A,
        f_a.data(), f_b.data(), f_c.data(),
        f_d.data(), f_e.data(), f_h.data(),
        g_a.data(), g_b.data(), g_c.data(),
        g_d.data(), g_e.data(), g_h.data(),
        rho, abs_tol, rel_tol, max_iter,
        verbose, adaptive_rho, /*gap_stop=*/1,
        beta, y_out.data(), l_out.data(),
        optval, final_iter
    );

    return status;
}
