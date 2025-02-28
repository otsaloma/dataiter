# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Dataiter'
copyright = '2020–2024 Osmo Salomaa'
author = 'Osmo Salomaa'

# The full version, including alpha/beta/rc tags
import dataiter
release = dataiter.__version__


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'output']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    'navigation_depth': 3,
}

html_context = {
    'display_github': True,
}

rst_prolog = """
:github_url: https://github.com/otsaloma/dataiter
"""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [
    '_static',
    'comparison/prism.css',
    'comparison/prism.js',
    'comparison/comparison.html',
]

def setup(app):
    # Build comparison/comparison.html. Note that readthedocs.org doesn't
    # run the Makefile, so anything there doesn't help in production.
    # https://github.com/readthedocs/readthedocs.org/issues/2276#issuecomment-231899567
    import subprocess
    from pathlib import Path
    cwd = Path(__file__).parent.resolve() / 'comparison'
    subprocess.run([sys.executable, 'build.py'], cwd=cwd, check=True)
