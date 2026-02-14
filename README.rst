===========
Pinball-QR
===========


.. image:: https://img.shields.io/pypi/v/pinball-qr.svg
        :target: https://pypi.python.org/pypi/pinball-qr

.. image:: https://github.com/mhoward2718/pinball-qr/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/mhoward2718/pinball-qr/actions/workflows/ci.yml

.. image:: docs/pinball_logo.png


The fastest and most accurate methods for quantile regression, now in python.

* Free software: MIT license
* Documentation: https://mhoward2718.github.io/pinball-qr/


Features
--------

* **sklearn-compatible API** — ``QuantileRegressor`` works with pipelines,
  cross-validation, and all the usual sklearn tooling.
* **Multiple solvers** — Barrodale-Roberts simplex (``br``), Frisch-Newton
  interior point (``fn``/``fnb``), preprocessed Frisch-Newton (``pfn``),
  LASSO-penalised (``lasso``), and ADMM via POGS (``pogs``).
* **Statistical inference** — IID, NID, and kernel standard errors, rank
  inversion confidence intervals, and bootstrap methods.
* **Nonparametric quantile regression** — optimal quantization via
  Competitive Learning Vector Quantization (CLVQ).

Credits & Acknowledgements
--------------------------

Pinball-QR is authored by **Michael Howard**.

This project builds on the work of several researchers and open-source projects:

**quantreg** (R package)
    The Fortran solvers (``rqbr``, ``rqfn``, ``rqfnb``) are ported from the
    `quantreg <https://cran.r-project.org/package=quantreg>`_ R package by
    **Roger Koenker**, with contributions from Stephen Portnoy, Pin Tian Ng,
    Blaise Melly, Achim Zeileis, and others. See Koenker, R. (2005)
    *Quantile Regression*, Cambridge University Press.

**POGS** (Proximal Operator Graph Solver)
    The ADMM-based solver uses vendored C++ source from
    `POGS <https://github.com/foges/pogs>`_ by **Christopher Fougner**
    (MIT license). See Fougner & Boyd (2018), *Parameter Selection and
    Pre-Conditioning for a Graph Form Solver*, in Emerging Applications of
    Control and Systems Theory, Springer.

**QuantifQuantile** (R package)
    The nonparametric quantile regression via optimal quantization is based on
    the `QuantifQuantile <https://cran.r-project.org/package=QuantifQuantile>`_
    R package by **Isabelle Charlier**, **Davy Paindaveine**, and
    **Jérôme Saracco**. See Charlier, Paindaveine & Saracco (2015),
    *Estimation of Conditional Quantiles using Optimal Quantization*.
