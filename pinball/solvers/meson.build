py = import('python').find_installation(pure: false)

# Backward-compatibility shim modules
python_sources = [
  '__init__.py',
  'base.py',
  'br.py',
  'fnb.py',
  'lasso.py',
  'pfn.py',
]

py.install_sources(
  python_sources,
  subdir: 'pinball/solvers',
)
