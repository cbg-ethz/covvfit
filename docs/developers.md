# For developers

## Installation

Clone the repository using

```bash
$ git clone git@github.com:cbg-ethz/covvfit.git
```

Then, create a new Python environment, e.g., using [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html):
```bash
$ micromamba create -n covvfit -c conda-forge python=3.11
```

Install the package in editable mode together with the developer utilities:

```bash
$ pip install poetry
$ poetry install --with dev
$ pre-commit install
```

## Testing

We use [Pytest](https://docs.pytest.org/) to write the unit tests. The unit tests are stored in `tests/` directory.
After the installation you can verify whether the unit tests are passed by running

```bash
$ pytest tests
```


## Documentation

We write the documentation using [Mkdocs](https://www.mkdocs.org/) in the `docs/` directory.
Apart from tutorials, we generate the API description directly from function docstrings.

Use

```bash
$ mkdocs build
```

to see whether the documentation is built properly.
Alternatively, to check the documentation in the interactive mode, use

```bash
$ mkdocs serve
```