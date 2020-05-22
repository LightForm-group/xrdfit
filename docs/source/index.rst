xrdfit documentation
======================

``xrdfit`` is a Python package for fitting the diffraction peaks in synchrotron X-ray diffraction (SXRD) and XRD spectra. It is intended as an easy to use tool for the quick analysis of individual and overlapping lattice plane peaks, to quantify the peak positions and profiles. ``xrdfit`` uses the Python package `lmfit <https://lmfit.github.io/lmfit-py/>`_ for the underlying fitting. Features are included for selecting different 'cakes' of data and automating fitting over many spectra, to enable tracking of peaks as they shift throughout an experiment. ``xrdfit`` is designed to be used by experimental researchers who need to process SXRD spectra but do not have a detailed knowledge of programming or fitting.


Installation
==============

``xrdfit`` is compatible with Python 3.6+.

Use :command:`pip` to install the latest stable version of ``xrdfit``:

.. code-block:: console

   pip install xrdfit

The current development version is available on `github
<https://github.com/LightForm-group/xrdfit>`__. To install use:

.. code-block:: console

   git clone --branch develop https://github.com/LightForm-group/xrdfit
   cd xrdfit
   python -m pip install . 


Getting started
================

This documentation is primarily an API reference, auto-generated from the docstrings in the source code. 

The primary source of documentation for new users is a series of tutorial Jupyter Notebooks which are included with the source code. 

The source and notebooks are available on the project’s GitHub page: `<https://github.com/LightForm-group/xrdfit>`_


Acknowledgements
=================

This project was developed at the `University of Manchester <https://www.manchester.ac.uk/>`_ with funding from the UK's Engineering and Physical Sciences Research Council (EPSRC) `LightForm <https://lightform.org.uk/>`_ grant: `(EP/R001715/1) <https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/R001715/1>`_.


API Reference
===============

.. toctree::
   :maxdepth: 3

   modules

* :ref:`genindex`

