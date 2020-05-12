# CONTRIBUTING

## Code style

* Python Code follows [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* C++ Code follows the existing code's style.

## Development

### Setup pip in Editable Mode

```bash
pip install --editable .
```

### Install Additional Requirements for Development

```bash
pip install -r requirements-dev.txt
```

## Testing

### Run tests

We use [pytest](https://docs.pytest.org/) to keep ika healthy. Make sure you follow the steps below, before and after your pull requests:

```bash
python install .
cd tests
pytest -r *.py
```

The test data is located in [here](https://github.com/scikit-ika/recurrent-data) and is automatically downloaded while running the tests.

## Sphinx documentation

Documentation follows the [NumPy/SciPy guidelines](https://numpydoc.readthedocs.io/en/latest/format.html) and you can find an example [here](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html).

The documentation page is hosted in the **gh-pages** branch. It is updated whenever a new version is released to PyPI.

```bash
cd docs
make html
```

See `docs/documentation.rst` for an example on how to add a new module.
