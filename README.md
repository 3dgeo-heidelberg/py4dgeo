# Welcome to GeoLib4d

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ssciwr/geolib4d/CI)](https://github.com/ssciwr/geolib4d/actions?query=workflow%3ACI)
[![PyPI Release](https://img.shields.io/pypi/v/geolib4d.svg)](https://pypi.org/project/geolib4d)
[![Documentation Status](https://readthedocs.org/projects/geolib4d/badge/)](https://geolib4d.readthedocs.io/)
[![codecov](https://codecov.io/gh/ssciwr/geolib4d/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/geolib4d)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_geolib4d&metric=alert_status)](https://sonarcloud.io/dashboard?id=ssciwr_geolib4d)

# Prerequisites

Building GeoLib4d requires the following software installed:

* A C++17-compliant compiler
* CMake `>= 3.9`
* Doxygen (optional, documentation building is skipped if missing)

* Python `>= 3.6` for building Python bindings


# Building GeoLib4d

The following sequence of commands builds GeoLib4d.
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

# Documentation

GeoLib4d provides a Sphinx-based documentation, that can
be browsed [online at readthedocs.org](https://geolib4d.readthedocs.io).

