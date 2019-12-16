Docs README
-------------

The documentation for xrdfit is generated using the Sphinx Python package and built and hosted at Read The Docs.

Documentation is written in reStructuredText format and compiled by the Sphinx package into a HTML source. At compilation time Sphinx also scrapes docstrings from the Python scripts and adds these to the documentation.


Requrements to compile documentation
======================================

Install: Sphinx (http://www.sphinx-doc.org/en/master/) - (conda install sphinx/pip install sphinx)


Compiling documentation
=========================

To recompile the documentation after a change, from the docs folder use the commands::

sphinx-build -b html ./source ./html

If the structure of the Python scripts is significantly changed, api-doc can autogenerate the API documentation with the command::

sphinx-apidoc -o ./source ../xrdfit

The HTML output should not be committed to the repository, this is just a local preview version. Each time the repository is pushed to GitHub the docs are rebuilt from source at: https://xrdfit.readthedocs.io/en/latest/