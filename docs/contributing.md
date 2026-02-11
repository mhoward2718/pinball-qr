# Contributing

Contributions are welcome!  Here's how to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourorg/pinball.git
cd pinball

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]" --no-build-isolation
```

## Dependencies

The Fortran extensions require a Fortran compiler.  On macOS:

```bash
brew install gcc
```

On Ubuntu/Debian:

```bash
sudo apt install gfortran
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pinball --cov-report=html

# Run sklearn check_estimator compliance
pytest tests/test_sklearn_compat.py -v
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings in NumPy style
- Keep lines under 88 characters (Black default)

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/ -v`)
5. Commit your changes (`git commit -m "Add my feature"`)
6. Push to your fork (`git push origin feature/my-feature`)
7. Open a Pull Request

## Adding a New Solver

The solver architecture uses the **Open/Closed Principle** — adding a
new solver requires no changes to existing code:

1. Create a new module in `pinball/linear/solvers/` (e.g. `my_solver.py`)
2. Subclass `BaseSolver` and implement `_solve_impl()`
3. Register the solver in `pinball/linear/solvers/__init__.py`:

```python
from pinball.linear.solvers.my_solver import MySolver
register_solver("my_method", MySolver)
```

4. Add tests in `tests/test_solvers.py`
5. Update the documentation

## Reporting Issues

Please include:

- Python version (`python --version`)
- NumPy/SciPy/sklearn versions (`pip list | grep -E "numpy|scipy|scikit"`)
- Operating system
- Minimal reproducible example
- Full traceback

## License

By contributing, you agree that your contributions will be licensed
under the MIT License.
