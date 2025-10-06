# Configuration file for the Sphinx documentation builder.

project = 'FluxGraph'
copyright = '2025, Alpha Networks'
author = 'Ihtesham Jahangir'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
