# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "geolib4d"
copyright = "2020, Dominic Kempf"
author = "Dominic Kempf"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Breathe Configuration: Breathe is the bridge between the information extracted
# from the C++ sources by Doxygen and Sphinx.
breathe_projects = {}
breathe_default_project = "geolib4d"

# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get("READTHEDOCS", None) == "True"

# Implement build logic on RTD servers
if read_the_docs_build:
    cwd = os.getcwd()
    os.makedirs("build-cmake", exist_ok=True)
    builddir = os.path.join(cwd, "build-cmake")
    subprocess.check_call(
        "cmake -DBUILD_DOCS=ON -DBUILD_TESTING=OFF -DBUILD_PYTHON=OFF ../..".split(),
        cwd=builddir,
    )
    subprocess.check_call("cmake --build . --target doxygen".split(), cwd=builddir)
    breathe_projects["geolib4d"] = os.path.join(builddir, "doc", "xml")
