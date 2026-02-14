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
cxx = meson.get_compiler('cpp')
if host_machine.system() == 'darwin'
  blas_lapack = [dependency('Accelerate', required: true)]
elif host_machine.system() == 'windows'
  blas_lapack = [dependency('openblas', required: true)]
else
  # 1. pkg-config / cmake: dependency('openblas')
  _openblas_dep = dependency('openblas', required: false)
  if _openblas_dep.found()
    blas_lapack = [_openblas_dep]
  else
    # 2. pkg-config: dependency('blas') + dependency('lapack')
    _blas_dep = dependency('blas', required: false)
    _lapack_dep = dependency('lapack', required: false)
    if _blas_dep.found() and _lapack_dep.found()
      blas_lapack = [_blas_dep, _lapack_dep]
    else
      # 3. Direct library search (manylinux2014 has no .pc files)
      blas_lapack = [
        cxx.find_library('openblas', required: true),
      ]
    endif
  endif
endif

# ── Build shared library ──────────────────────────────────
_pogs_native = shared_library(
  '_pogs_native',
  [pogs_sources, adapter_sources],
  include_directories: pogs_inc,
  dependencies: blas_lapack,
  cpp_args: ['-std=c++20', '-DQR_POGS_BUILDING', '-DPOGS_BUILDING_SHARED'],
  install: true,
  install_dir: py.get_install_dir() / 'pinball',
)
