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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import recommonmark
from recommonmark.transform import AutoStructify
import os
import sys
from sphinx.domains import Domain
import datetime
# -- Project information -----------------------------------------------------

project = 'Sentence-Transformers'
copyright = str(datetime.datetime.now().year)+', Nils Reimers'
author = 'Nils Reimers'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'recommonmark',
    'sphinx_markdown_tables'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'nr_examples']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = ["_themes"]

html_theme_options = {
    'logo_only': True,
    'canonical_url': "https://www.sbert.net"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

html_js_files = [
    'js/custom.js',
]

html_show_sourcelink = False
html_context = {
  'display_github': True,
  'github_user': 'UKPLab',
  'github_repo': 'sentence-transformers',
  'github_version': 'master/',
}

html_logo = 'img/logo.png'
html_favicon = 'img/favicon.ico'

autoclass_content = 'both'



class GithubURLDomain(Domain):
    """
    Resolve .py links to their respective Github URL
    """

    name = "githuburl"
    ROOT = "https://github.com/UKPLab/sentence-transformers/tree/master"

    def resolve_any_xref(self, env, fromdocname, builder, target, node, contnode):
        if (target.endswith('.py') or target.endswith('.ipynb')) and not target.startswith('http'):
            from_folder = os.path.dirname(fromdocname)
            contnode["refuri"] = "/".join([self.ROOT, from_folder, target])
            return [("githuburl:any", contnode)]
        return []



def setup(app):
    app.add_domain(GithubURLDomain)
    app.add_config_value('recommonmark_config', {
            #'url_resolver': lambda url: github_doc_root + url,
            'auto_toc_tree_section': 'Contents',
            }, True)
    app.add_transform(AutoStructify)
