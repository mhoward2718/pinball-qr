# Build the POGS-based quantile regression native library.
#
# Compiles vendored POGS source (CPU only) and our thin
# qr_pogs adapter into a shared library that Python loads via ctypes.

py = import('python').find_installation(pure: false)

# ── POGS vendored sources ──────────────────────────────────
pogs_sources = files(
  'pogs/cpu/pogs.cpp',
  'pogs/cpu/matrix/matrix_dense.cpp',
  'pogs/cpu/matrix/matrix_sparse.cpp',
  'pogs/cpu/projector/projector_cgls.cpp',
  'pogs/cpu/projector/projector_direct_dense.cpp',
  'pogs/interface_c/pogs_c.cpp',
)

# ── Our adapter ────────────────────────────────────────────
adapter_sources = files('qr_pogs.cpp')

# ── Include directories for POGS headers ───────────────────
pogs_inc = include_directories(
  '.',                     # qr_pogs.h
  'pogs/include',          # pogs.h, prox_lib.h, matrix/, projector/
  'pogs/cpu/include',      # equil_helper.h, cgls.h, gsl/*
  'pogs/interface_c',      # pogs_c.h
)

# ── BLAS / LAPACK ──────────────────────────────────────────────────
# On macOS, use the Accelerate framework (always available, no MKL needed).
# On Linux, fall back to pkg-config / cmake discovery.
# On Windows, CI installs OpenBLAS and sets PKG_CONFIG_PATH.
if host_machine.system() == 'darwin'
  blas_lapack = [dependency('Accelerate', required: true)]
elif host_machine.system() == 'windows'
  blas_lapack = [dependency('openblas', required: true)]
else
  blas_lapack = [
    dependency('blas', required: true),
    dependency('lapack', required: true),
  ]
endif

# ── Build shared library ──────────────────────────────────
_pogs_native = shared_library(
  '_pogs_native',
  [pogs_sources, adapter_sources],
  include_directories: pogs_inc,
  dependencies: blas_lapack,
  cpp_args: ['-std=c++20', '-DQR_POGS_BUILDING'],
  install: true,
  install_dir: py.get_install_dir() / 'pinball',
)
