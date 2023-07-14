import sphinx_rtd_theme


""" Project information """
project = 'hbmep'
copyright = '2023, hbmep-authors'
author = 'hbmep-authors'
release = '0.0.1'

""" General configuration """
extensions = []

templates_path = ['_templates']
exclude_patterns = []



""" Options for HTML output """
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ['_static']
