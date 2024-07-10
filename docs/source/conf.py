from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("hbmep").version
except DistributionNotFound:
    __version__ = "unknown version"


""" Project information """
project = 'hbmep'
copyright = '2023-2024, hbmep authors'
version = __version__
release = __version__

""" General configuration """
extensions = [
    'sphinxcontrib.bibtex',
    'myst_nb',
    'sphinx_copybutton',
]

bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'author_year'

templates_path = ['_templates']
exclude_patterns = []

""" Options for HTML output """
# HTML theme
html_theme = "sphinx_book_theme"
html_context = {
    # ...
    "default_mode": "dark"
}
# html_copy_source = True
# html_show_sourcelink = True
# html_sourcelink_suffix = ""
html_title = f"hbmep v{__version__}"
# html_favicon = "_static/favicon.png"
# html_static_path = ["_static"]
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/hbmep/hbmep",
    "repository_branch": "main",
    # "launch_buttons": {
    #     "binderhub_url": "https://mybinder.org",
    #     "notebook_interface": "classic",
    # },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
nb_execution_mode = "off"
nb_execution_timeout = -1