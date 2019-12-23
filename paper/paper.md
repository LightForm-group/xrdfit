---
title: 'xrdfit: A Python package for fitting SXRD diffraction spectra'
tags:
  - Python
  - physics
  - crystallography
  - material structure
authors:
  - name: Peter Crowther
    orcid: 0000-0002-5430-6924
    affiliation:  1
  - name: Christopher Daniel
	orcid: 0000-0002-5574-6833
    affiliation: 1
affiliations:
 - name: Univeristy of Manchester
   index: 1
date: 16 December 2019
bibliography: paper.bib

---

# Summary

100 words on xrd and material structure

``xrdfit`` is an Python package for fitting the peaks found in shallow x-ray 
diffraction spectra. It is designed to be an easy to use tool for quick analysis of
spectra. Features are included for automating fitting over many spectra to enable
tracking of peaks as they shift through the experiment. Some basic Materials anlysis
algorithms are added which allow converstion of the peak positions to material properties
like strain. ``xrdfit`` uses the Python package ``lmfit`` for the underlying fitting [@Newville2014].

``xrdfit`` is designed to be used by experimental researchers who need to 
process SXRD spectra but do not have a detailed knowledge of programming or
fitting. It has been used for the analysis of data presented in [insert paper here].
We hope that its public release will allow other researchers to benefit from 
fast data fitting, reducing the effort required to do basic analysis of their
experimental data.

# Acknowledgements

We acknowledge funding from 

# References