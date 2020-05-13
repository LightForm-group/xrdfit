---
title: 'xrdfit: A Python package for fitting synchrotron X-ray diffraction spectra'
tags:
  - Python
  - crystallography
  - x-ray diffraction
  - synchrotron
  - material structure
  - peak fitting
  - data analysis

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
date: 12 May 2020
bibliography: paper.bib

---

# Summary

The evolution of peak profiles in synchrotron X-ray diffraction (SXRD) data can tell us how the internal crystallographic structures of metals change during applied heating, high temperature straining and cooling cycles [@Daniel_2019; @Stark_2015; @Canelo_Yubero_2016; @Hu_2017], which is invaluable information used to improve industrial processing routes [@Salem_2008]. The experiment requires a beamline, such as Diamond Light Source [@Diamond_2020], to produce a high energy X-ray beam and illuminate a polycrystalline sample [@Daniel_2019]. The results are recorded in the form of time-resolved diffraction pattern rings, which are converted into a spectra of intensity peaks versus two-theta angle for a given direction [@Filik_2017; @Ashiotis_2015; @Hammersley_1996]. However, since many intensity profiles are collected during each experiment, with detectors recording at speeds of up to 250 Hz [@Diamond2_2020], fitting each of the individual lattice plane peaks can take a long time using current available software [@Basham_2015; @Merkel_2015]. It is also difficult to distinguish the evolution of any overlapping peaks for multi-phase materials. Therefore, a faster and more robust Python package has been developed to fit the evolution of multiple and overlapping peaks for datasets containing many thousands of SXRD patterns.

``xrdfit`` is a Python package for fitting the diffraction peaks in SXRD (and XRD) spectra. It is intended as an easy to use tool for the quick analysis of individual and overlapping lattice plane peaks, to discern the peak position and profile. ``xrdfit`` uses the Python package ``lmfit`` [@Newville_2014] for the underlying fitting. Features are included for selecting different 'cakes' of data and automating fitting over many spectra, to enable tracking of peaks as they shift throughout the experiment. By analysing how different lattice plane peaks change during simulated processing, the transformation and micromechanical behaviour of the material can be understood.

``xrdfit`` is designed to be accessible for all researchers who need to process SXRD (and XRD) spectra and so does not require a detailed knowledge of programming or fitting. It has been used for the analysis of data taken during the hot deformation of a two-phase titanium alloy, which is presented in an article currently in press [@Daniel_2019], and will be used for future studies investigating the high temperature processing of metals in our research group. We hope that its public release will allow other researchers to benefit from fast data fitting, reducing the effort required to analyse their experimental data.

# Acknowledgements

We acknowledge funding from the UK's Engineering and Physical Sciences Research Council (EPSRC) via LightForm (EP/R001715/1). We thank Oliver Buxton for his comments on a pre-release version of xrdfit.

# References