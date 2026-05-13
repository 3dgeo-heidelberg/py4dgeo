# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import py4dgeo
import subprocess
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
doc_dir = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(doc_dir, ".."))
python_package_src = os.path.join(repo_root, "src")
sys.path.insert(0, python_package_src)

# We need to be able to locate data files to include Jupyter notebooks
os.environ["XDG_DATA_DIRS"] = os.path.abspath("../tests/data")

# -- Project information -----------------------------------------------------

project = "py4dgeo"
copyright = "2021, Scientific Software Center, Heidelberg University"
author = "Dominic Kempf"
release = py4dgeo.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "breathe",
    "myst_nb",
    "nbsphinx_link",
]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}

# MyST parser configuration
myst_enable_extensions = [
    "html_image",
]

# Allow errors in notebooks to avoid connection issues
nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
html_extra_path = ["img"]

# Breathe Configuration: Breathe is the bridge between the information extracted
# from the C++ sources by Doxygen and Sphinx.
breathe_projects = {
    "py4dgeo": os.path.join(repo_root, "build", "doc", "xml"),
}
breathe_default_project = "py4dgeo"
