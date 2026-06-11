# Release checklist

This repository is configured for PyPI-oriented Python releases.

## Before release

Run locally:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build --sdist
python -m twine check dist/*
mkdocs build --strict
```

To build PyPI-compatible wheels locally, run cibuildwheel. Linux wheel builds
require Docker:

```bash
python -m cibuildwheel
python -m twine check wheelhouse/*
```

## GitHub checks

Pull requests run:

- tests across supported Python versions,
- source distribution builds,
- manylinux wheel builds via cibuildwheel,
- pip installation from the built wheel,
- pip installation from the built source distribution,
- strict documentation builds.

## Publishing to PyPI

The `Publish Python package` workflow runs on GitHub releases and uses PyPI
trusted publishing. It publishes the source distribution and CPython wheels for
Linux, macOS, and Windows.

Repository maintainers need to configure a PyPI trusted publisher for this
repository and the `pypi` GitHub environment before the first release.

After the release workflow succeeds, users can install with:

```bash
python -m pip install addivortes
```
