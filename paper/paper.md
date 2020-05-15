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
  - name: Christopher S. Daniel
    orcid: 0000-0002-5574-6833
    affiliation: 1
affiliations:
 - name: Univeristy of Manchester
   index: 1
date: 12 May 2020
bibliography: paper.bib

---

# Summary

The evolution of peak profiles in synchrotron X-ray diffraction (SXRD) data can tell us how the internal crystallographic structures of metals change during applied heating, high temperature straining and cooling cycles [@Daniel_2019; @Stark_2015; @Canelo_Yubero_2016; @Hu_2017], which is invaluable information used to improve industrial processing routes [@Salem_2008]. The experiment requires a beamline, such as Diamond Light Source [@Diamond_2020], to produce a high energy X-ray beam and illuminate a polycrystalline sample [@Daniel_2019]. The results are recorded in the form of time-resolved diffraction pattern rings, which are converted into a spectra of intensity peaks versus two-theta angle for a given direction [@Filik_2017; @Ashiotis_2015; @Hammersley_1996]. However, since many intensity profiles are collected during each experiment, with detectors recording at speeds of up to 250 Hz [@Diamond2_2020], fitting each of the individual lattice plane peaks can take a long time using current available software [@Basham_2015; @Merkel_2015]. Distinguishing the evolution of any overlapping peaks, for multi-phase materials, can also be difficult and time-consuming. Therefore, a faster and more robust Python package has been developed to fit the evolution of multiple and overlapping peaks for datasets containing many thousands of SXRD patterns.

``xrdfit`` is a Python package for fitting the diffraction peaks in SXRD (and XRD) spectra. It is intended as an easy to use tool for the quick analysis of individual and overlapping lattice plane peaks, to discern the peak position and profile. The features of  ``xrdfit`` are shown schematically in \autoref{fig:figure1}. ``xrdfit`` uses the Python package ``lmfit`` [@Newville_2014] for the underlying fitting. Features are included for selecting different 'cakes' of data and automating fitting over many spectra, to enable tracking of peaks as they shift throughout the experiment. By analysing how different lattice plane peaks change during simulated processing, as can be seen in \autoref{fig:figure2}, the transformation and micromechanical behaviour of the material can be understood.

![A schematic representing the features of xrdfit and the analysis of a synchrotron X-ray diffraction (SXRD) pattern, showing: (a) a polar plot of the caked intensity data; (b) an option for selecting different cakes and merging caked datasets; (c) adjustable peak bounds (grey) and adjustable maxima and maxima bounds (red and green) for constraining the peak fit; (d) an example fit of multiple and overlapping peaks.\label{fig:figure1}](figure1.eps)

``xrdfit`` is designed to be accessible for all researchers who need to process SXRD (and XRD) spectra and so does not require a detailed knowledge of programming or fitting. The package has been used for the analysis of data taken during the hot deformation of a two-phase titanium alloy, which is presented in an article currently in press [@Daniel_2019], and will be used for future studies investigating the high temperature processing of metals in our research group. We hope that its public release will allow other researchers to benefit from fast data fitting, reducing the effort required to analyse their experimental data.

![An example analysis of a two-phase titanium (Ti-6Al-4V) alloy during high temperature and high strain rate deformation, showing characteristic shifts of the $\alpha$ (0002) peak centre. The shifts of the peak can be used to calculate elastic lattice straining in the hexagonal close-packed (hcp) lattice, as well as measure the thermal contraction on cooling.\label{fig:figure2}](figure2.eps)

# Acknowledgements

We acknowledge funding from the UK's Engineering and Physical Sciences Research Council (EPSRC) via LightForm (EP/R001715/1). We thank Oliver Buxton for his comments on a pre-release version of xrdfit.

# References