xrdfit documentation
======================

``xrdfit`` is a Python package for fitting the diffraction peaks in synchrotron X-ray diffraction (SXRD) and XRD spectra. It is intended as an easy to use tool for the quick analysis of individual and overlapping lattice plane peaks, to quantify the peak positions and profiles. ``xrdfit`` uses the Python package `lmfit <https://lmfit.github.io/lmfit-py/>`_ for the underlying fitting. Features are included for selecting different 'cakes' of data and automating fitting over many spectra, to enable tracking of peaks as they shift throughout the experiment. ``xrdfit`` is designed to be used by experimental researchers who need to process SXRD spectra but do not have a detailed knowledge of programming or fitting.


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


Comparison to other peak fitting tools
========================================

DAWN
-----

DAWN `<https://dawnsci.org/>`_ is a multipurpose framework, developed to deal with analysis of a whole range of
datasets measured from beamline experiments. It is the most comparable tool we have come across in in terms of its
ability to do simple fits like those done by `xrdfit`.

Testing the fitting of a clear single peak for a dataset of 1000 spectra we find that DAWN typically takes 1-2 seconds
per image and provides only limited information about the result of the fit. It also only appears to allow fitting of
a single peak at a time, fitting multiple peaks would mean running the same analysis each time for each peak.
We found that the program crashed when attempting to load more than 1000 spectra for analysis.

For `xrdfit`, fitting of the same peak is on the order of 0.05 seconds per spectra. It is also possible to set up
an analysis to fit multiple peaks simultaneously which greatly speeds the analysis of a full spectrum. Fitting is
much more interactive, meaning that it is possible to more quickly determine if fits are correct and modify them if
necessary. `xrdfit` can also easily deal with large datasets.

MAUD
-----

MAUD can be used to fit peaks in diffraction spectra but uses a Rietveld refinement method to match a model of the
material properties and the beamline setup to the data. This method provides different information about the material
but because MAUD uses a model of the unit cell to fit the data, there is some averaging of the overall fit and a larger
error in the individual peak position. In addition, automation of fitting in MAUD is complex and depends on identifying
peaks using the user interface which can't be used as part of a reproducible analysis.


Acknowledgements
=================

This project was developed at the `University of Manchester <https://www.manchester.ac.uk/>`_ with funding from the UK's Engineering and Physical Sciences Research Council (EPSRC) `LightForm <https://lightform.org.uk/>`_ grant: `(EP/R001715/1) <https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/R001715/1>`_.


API Reference
===============

.. toctree::
   :maxdepth: 3

   modules

* :ref:`genindex`

