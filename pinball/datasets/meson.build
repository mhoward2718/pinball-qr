py = import('python').find_installation(pure: false)

python_sources = [
  '__init__.py',
  '_engel.py',
  '_barro.py',
]

py.install_sources(
  python_sources,
  subdir: 'pinball/datasets',
)

# Install bundled CSV data files
py.install_sources(
  ['engel.csv', 'barro.csv'],
  subdir: 'pinball/datasets',
)
