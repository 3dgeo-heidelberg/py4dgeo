# Welcome to py4dgeo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ssciwr/py4dgeo/CI)](https://github.com/ssciwr/py4dgeo/actions?query=workflow%3ACI)
[![PyPI Release](https://img.shields.io/pypi/v/py4dgeo.svg)](https://pypi.org/project/py4dgeo)
[![Documentation Status](https://readthedocs.org/projects/py4dgeo/badge/)](https://py4dgeo.readthedocs.io/)
[![codecov](https://codecov.io/gh/ssciwr/py4dgeo/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/py4dgeo)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_py4dgeo&metric=alert_status)](https://sonarcloud.io/dashboard?id=ssciwr_py4dgeo)

py4dgeo is a C++ library with Python bindings for change detection in 4D point cloud data.
It is currently *under active development*.

# Prerequisites

Building py4dgeo requires the following software installed:

* A C++17-compliant compiler
* CMake `>= 3.9`
* Doxygen (optional, documentation building is skipped if missing)
* Python `>= 3.6` for building Python bindings

In order to contribute to the development of py4dgeo, you should additionally
install the following tools:

* [Pre-commit](https://pre-commit.com/) and enable it by doing `pre-commit install` in the repository.

# Building py4dgeo

The following sequence of commands builds py4dgeo.
It assumes that your current working directory is the top-level directory
of the freshly cloned repository:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

The build process can be customized with the following CMake variables,
which can be set by adding `-D<var>={ON, OFF}` to the `cmake` call:

* `BUILD_TESTING`: Enable building of the test suite (default: `ON`)
* `BUILD_DOCS`: Enable building the documentation (default: `ON`)
* `BUILD_PYTHON`: Enable building the Python bindings (default: `ON`)

Additionally, py4dgeo provides a Docker image that allows to explore
the library using JupyterLab. The image can be locally built and run with
the following commands:

```
docker build -t py4dgeo:latest .
docker run -t -p 8888:8888 py4dgeo:latest
```

# Documentation

py4dgeo provides a Sphinx-based documentation, that can
be browsed [online at readthedocs.org](https://py4dgeo.readthedocs.io).
