/*
 * qr_pogs.h — C interface for quantile regression via POGS ADMM.
 *
 * This is a thin adapter that exposes a single function for solving
 * quantile regression problems.  It hides the full generality of the
 * POGS graph-form API behind a purpose-built interface.
 *
 * The pinball loss is:
 *   rho_tau(u) = 0.5 * |u| + (tau - 0.5) * u
 *
 * Standard QR minimises rho_tau(y - Xb).  In POGS graph-form with
 * y_var = Xb this becomes:
 *   f_i: FunctionObj(kAbs, a=1, b=y_i, c=0.5, d=0.5-tau, e=0)
 *   g_j: FunctionObj(kZero)
 *
 * Copyright (c) 2026 pinball contributors.  Apache-2.0 (POGS) + MIT (adapter).
 */

#ifndef QR_POGS_H
#define QR_POGS_H

#include <stddef.h>

#ifdef _WIN32
  #ifdef QR_POGS_BUILDING
    #define QR_POGS_API __declspec(dllexport)
  #else
    #define QR_POGS_API __declspec(dllimport)
  #endif
#else
  #define QR_POGS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solve a quantile regression problem via POGS ADMM.
 *
 *   minimize  sum_i  rho_tau( y_i  -  x_i' beta )
 *
 * @param m          Number of observations.
 * @param n          Number of predictors (columns of A).
 * @param A          Design matrix, row-major, m x n.
 * @param y          Response vector, length m.
 * @param tau        Quantile level in (0, 1).
 * @param abs_tol    Absolute tolerance  (e.g. 1e-4).
 * @param rel_tol    Relative tolerance  (e.g. 1e-4).
 * @param max_iter   Maximum ADMM iterations (e.g. 2500).
 * @param verbose    Verbosity (0 = silent).
 * @param adaptive_rho  1 = adaptive, 0 = fixed.
 * @param rho        Initial penalty parameter (e.g. 1.0).
 *
 * @param[out] beta       Coefficient vector, length n.
 * @param[out] optval     Optimal objective value.
 * @param[out] final_iter Number of iterations taken.
 *
 * @return 0 on success, non-zero on error.
 */
QR_POGS_API int qr_pogs_solve(
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
    unsigned int *final_iter
);

#ifdef __cplusplus
}
#endif

#endif /* QR_POGS_H */
