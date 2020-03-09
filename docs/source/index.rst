xrdfit documentation
======================

``xrdfit`` is a Python package for fitting the peaks found in shallow x-ray diffraction spectra. It is designed to be an easy to use tool for quick analysis of spectra. Features are included for automating fitting over many spectra to enable tracking of peaks as they shift through the experiment. ``xrdfit`` uses the Python package `lmfit <https://lmfit.github.io/lmfit-py/>`_ for the underlying fitting. ``xrdfit`` is designed to be used by experimental researchers who need to process SXRD spectra but do not have a detailed knowledge of programming or fitting.


Installation
==============

``xrdfit`` is compatible with Python 3.6+.

Use :command:`pip` to install the latest stable version of ``xrdfit``:

.. code-block:: console

   pip install xrdfit

The current development version is available on `github
<https://github.com/LightForm-group/xrdfit>`__. Use :command:`git` and
:command:`python setup.py` to install it:

.. code-block:: console

   git clone https://github.com/LightForm-group/xrdfit
   cd xrdfit
   python setup.py install


Getting started
================

This documentation is primarily an API reference, auto-generated from the docstrings in the source code. 

The primary source of documentation for new users is a series of tutorial Jupyter Notebooks which are included with the source code. 

The source and notebooks are available at the project’s GitHub page: `<https://github.com/LightForm-group/xrdfit>`_


Acknowledgements
=================

This project was developed at the `University of Manchester <https://www.manchester.ac.uk/>`_ with funding from the EPSRC `LightForm <https://lightform.org.uk/>`_ grant: `(EP/R001715/1) <https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/R001715/1>`_.


API Reference
===============

.. toctree::
   :maxdepth: 3

   modules

* :ref:`genindex`

