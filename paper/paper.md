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

The evolution of peak profiles in synchrotron X-ray diffraction (SXRD) data can tell us how the internal crystallographic structures of metals change during applied heating, high temperature straining and cooling cycles `[@daniel_christopher_stuart_2019_3381183; @Stark2015; @Canelo-Yubero2016; @HU2017230]`, which is invaluable information  used to improve industrial processing routes `[@salem2008]`. The experiment requires a beamline, such as Diamond Light Source `[@DiamondLightSourceLtd2020]`, to produce a high energy X-ray beam and illuminate a polycrystalline sample `[@daniel_christopher_stuart_2019_3381183]`. The results are recorded in the form of time-resolved diffraction pattern rings, which are converted into a spectra of intensity peaks versus two-theta angle for a given direction `[@Filik2017; @Ashiotis2015; @Hammersley1996]`. However, since many intensity profiles are collected during each experiment, with detectors recording at speeds of up to 250 Hz `[@DiamondLightSourceLtdDetectors2020]`, fitting each of the individual lattice plane peaks can take a long time using current available software `[@Basham2015; @Merkel2015]`. It is also difficult to distinguish the evolution of any overlapping peaks in multi-phase materials. Therefore, a faster and more robust python script has been produced to fit the evolution of multiple and overlapping peaks for SXRD datasets containing many thousands of patterns.


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