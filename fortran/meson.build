# Build Fortran extensions via f2py
#
# All three Fortran solvers are compiled into a single extension module:
#   pinball._native  (importable as pinball._native.rqbr, .rqfnb, .rqfn)

py = import('python').find_installation(pure: false)

# Use f2py to generate wrapper and compile Fortran sources
fortran_sources = files(
  'rqbr.f',
  'rqfn.f',
  'rqfnb.f',
)

incdir_numpy = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
).stdout().strip()

incdir_f2py = run_command(
  py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true,
).stdout().strip()

# f2py runtime source — defines _PyFortran_Type and other symbols
f2py_source = incdir_f2py / 'fortranobject.c'

# ── BLAS / LAPACK (needed by rqfn, rqfnb, rqbr — they call dswap, dgemv, …) ──
if host_machine.system() == 'darwin'
  blas_lapack = [dependency('Accelerate', required: true)]
else
  blas_lapack = [
    dependency('blas', required: true),
    dependency('lapack', required: true),
  ]
endif

# Generate the f2py wrapper C source
f2py_gen = custom_target(
  '_nativemodule',
  input: fortran_sources,
  output: ['_nativemodule.c', '_native-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', '_native', '--lower',
            '--build-dir', '@OUTDIR@'],
)

# Build the extension module
py.extension_module(
  '_native',
  [f2py_gen, fortran_sources, f2py_source],
  include_directories: [incdir_numpy, incdir_f2py],
  dependencies: blas_lapack,
  install: true,
  subdir: 'pinball',
)
