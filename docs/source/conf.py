import os
import sys
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("hbmep").version
except DistributionNotFound:
    __version__ = "unknown version"

""" Project information """
project = 'hbmep'
copyright = '2023-2026, hbmep authors'
version = __version__
release = __version__

""" General configuration """
extensions = [
    'myst_nb',
    'sphinx_copybutton',
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = []

""" Options for HTML output """
html_theme = "sphinx_book_theme"
html_context = {
    "default_mode": "dark"
}
html_title = f"hbmep v{__version__}"
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/hbmep/hbmep",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
nb_execution_mode = "off"
nb_execution_timeout = -1
