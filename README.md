# Welcome to py4dgeo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ssciwr/py4dgeo/CI)](https://github.com/ssciwr/py4dgeo/actions?query=workflow%3ACI)
[![PyPI Release](https://img.shields.io/pypi/v/py4dgeo.svg)](https://pypi.org/project/py4dgeo)
[![Documentation Status](https://readthedocs.org/projects/py4dgeo/badge/)](https://py4dgeo.readthedocs.io/)
[![codecov](https://codecov.io/gh/ssciwr/py4dgeo/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/py4dgeo)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_geolib4d&metric=alert_status)](https://sonarcloud.io/dashboard?id=ssciwr_geolib4d)

py4dgeo is a C++ library with Python bindings for change detection in 4D point cloud data.
It is currently *under active development*.

## Prerequisites

Using py4dgeo requires the following software installed:

* Python `>= 3.6`

In order to build the package from source, the following tools are also needed.

* A C++17-compliant compiler
* CMake `>= 3.9`
* Doxygen (optional, documentation building is skipped if missing)

## Installing and using py4dgeo

The preferred way of installing `py4dgeo` is using `pip`.
### Using pip

`py4dgeo` can be installed using `pip`:

```
python -m pip install py4dgeo
```

### Building from source

The following sequence of commands is used to build `py4dgeo` from source:

```
git clone --recursive https://github.com/ssciwr/py4dgeo.git
cd py4dgeo
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

If you want to contribute to the library's development you should also install
its additional Python dependencies for testing and documentation building:

```
python -m pip install -r requirements-dev.txt`
```

The build directory contains a file `setup-pythonpath.sh` that you can use
to modify your `PYTHONPATH` during development, so that it includes the compiled
module, as well as the Python package from the source directory:

```
source build/setup-pythonpath.sh
```

### Using Docker

Additionally, `py4dgeo` provides a Docker image that allows to explore
the library using JupyterLab. The image can be locally built and run with
the following commands:

```
docker build -t py4dgeo:latest .
docker run -t -p 8888:8888 py4dgeo:latest
```
