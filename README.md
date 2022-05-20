# Welcome to py4dgeo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ssciwr/py4dgeo/CI)](https://github.com/ssciwr/py4dgeo/actions?query=workflow%3ACI)
[![PyPI Release](https://img.shields.io/pypi/v/py4dgeo.svg)](https://pypi.org/project/py4dgeo)
[![Documentation Status](https://readthedocs.org/projects/py4dgeo/badge/)](https://py4dgeo.readthedocs.io/)
[![codecov](https://codecov.io/gh/ssciwr/py4dgeo/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/py4dgeo)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_geolib4d&metric=alert_status)](https://sonarcloud.io/dashboard?id=ssciwr_geolib4d)

![logo](py4dgeo_logo_mini.png)

py4dgeo is a C++ library with Python bindings for change detection in 4D point cloud data.
It is currently *under active development*.

## Prerequisites

Using py4dgeo requires the following software installed:

* Python `>= 3.7`

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
python -m pip install --editable .
```

The `--editable` flag allows you to change the Python sources of `py4dgeo` without
reinstalling the package. To recompile the C++ source, please run `pip install` again.
In order to enable multi-threading on builds from source, your compiler toolchain
needs to support `OpenMP`.

If you want to contribute to the library's development you should also install
its additional Python dependencies for testing and documentation building:

```
python -m pip install -r requirements-dev.txt
```

### Using Docker

Additionally, `py4dgeo` provides a Docker image that allows to explore
the library using JupyterLab. The image can be locally built and run with
the following commands:

```
docker build -t py4dgeo:latest .
docker run -t -p 8888:8888 py4dgeo:latest
```
