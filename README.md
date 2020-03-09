[![GitHub Release] (https://img.shields.io/github/v/release/LightForm-group/xrdfit)
[![PyPI version](https://badge.fury.io/py/xrdfit.svg)](https://badge.fury.io/py/xrdfit)
[![Documentation Status](https://readthedocs.org/projects/xrdfit/badge/?version=latest)](https://xrdfit.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/203145007.svg)](https://zenodo.org/badge/latestdoi/203145007) 

Introduction
--------------

xrdfit is a Python module for fitting the peaks found in shallow x-ray diffraction spectra. It is designed to be an easy to use tool for quick analysis of spectra. Features are included for automating fitting over many spectra to enable tracking of peaks as they shift through the experiment. xrdfit uses the Python module lmfit for the underlying fitting. xrdfit is designed to be accessible for all researchers who need to process SXRD spectra and so does not require a detailed knowledge of programming or fitting.

Installation
-------------

To install as a Python module, type

`python -m pip install xrdfit`

from the root directory. 
For developers, you should install in linked .egg mode using

`python -m pip install -e .`

If you are using a Python virtual environment, you should activate this first before using the above commands.

Documentation
---------------
Documentation including an API reference is provided at: https://xrdfit.readthedocs.io/en/latest/

The majority of the documentation is provided as example driven interactive Jupyter notebooks. These are included along with the source code in the "tutorial notebooks" folder.
If this package was downloaded from pip, the source can be found on GitHub: https://github.com/LightForm-group/xrdfit


Required libraries
--------------------

This module uses the Python libraries:
* NumPy (https://numpy.org/)
* matplotlib (https://matplotlib.org/)
* pandas (https://pandas.pydata.org/)
* dill (https://pypi.org/project/dill/)
* tqdm (https://tqdm.github.io/)
* SciPy (https://www.scipy.org/)
* lmfit (https://lmfit.github.io/lmfit-py/)

The following libraries are required to use the tutorial documentation workbooks:
* Jupyter (https://jupyter.org/)
