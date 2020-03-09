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
date: 9 March 2020
bibliography: paper.bib

---


# Summary

The evolution of peak profiles in synchrotron X-ray diffraction (SXRD) data can tell us how the internal crystallographic structures of metals change during applied heating, high temperature straining and cooling cycles [@Daniel_2019; @Stark_2015; @Canelo_Yubero_2016; @Hu_2017], which is invaluable information  used to improve industrial processing routes [@Salem_2008]. The experiment requires a beamline, such as Diamond Light Source [@Diamond_2020], to produce a high energy X-ray beam and illuminate a polycrystalline sample [@Daniel_2019]. The results are recorded in the form of time-resolved diffraction pattern rings, which are converted into a spectra of intensity peaks versus two-theta angle for a given direction [@Filik_2017; @Ashiotis_2015; @Hammersley_1996]. However, since many intensity profiles are collected during each experiment, with detectors recording at speeds of up to 250 Hz [@Diamond2_2020], fitting each of the individual lattice plane peaks can take a long time using current available software [@Basham_2015; @Merkel_2015]. It is also difficult to distinguish the evolution of any overlapping peaks in multi-phase materials. Therefore, a faster and more robust Python package has been developed to fit the evolution of multiple and overlapping peaks for SXRD datasets containing many thousands of patterns.

``xrdfit`` is a Python package for fitting the peaks found in shallow x-ray diffraction spectra. It is intended as an easy to use tool for quick analysis of spectra. Features are included for automating fitting over many spectra to enable tracking of peaks as they shift through the experiment. ``xrdfit`` uses the Python package ``lmfit`` [@Newville_2014] for the underlying fitting.

``xrdfit`` is designed to be accessible for all researchers who need to process SXRD spectra and so does not require a detailed knowledge of programming or fitting. It has been used for the analysis of data presented in an article currently in preparation [@Daniel_2019] and will be used for future studies in our group. We hope that its public release will allow other researchers to benefit from fast data fitting, reducing the effort required to do basic analysis of their experimental data.

# Acknowledgements

We acknowledge funding from EPSRC grant LightForm (EP/R001715/1). We thank Oliver Buxton for his comments on a pre-release version of xrdfit.

# References