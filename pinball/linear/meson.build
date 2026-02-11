py = import('python').find_installation(pure: false)

python_sources = [
  '__init__.py',
  '_estimator.py',
  '_inference.py',
  '_bootstrap.py',
]

py.install_sources(
  python_sources,
  subdir: 'pinball/linear',
)

subdir('solvers')
