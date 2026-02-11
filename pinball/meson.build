# Install all Python source files for the pinball package
py = import('python').find_installation(pure: false)

python_sources = [
  '__init__.py',
  '_bootstrap.py',
  '_estimator.py',
  '_inference.py',
  '_typing.py',
]

py.install_sources(
  python_sources,
  subdir: 'pinball',
)

# Install subpackages
subdir('estimators')
subdir('linear')
subdir('solvers')
subdir('util')
subdir('datasets')
