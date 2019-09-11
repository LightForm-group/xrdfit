import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import PseudoVoigtModel
from lmfit import Model
from typing import List, Union
from abc import ABC


class PeakParams:
    def __init__(self, peak_center=None, center_min=None, center_max=None, sigma_min=None,
                 sigma_max=None, amplitude_min=None):
        """
        An object representing a description of a peak location.
        :param peak_center: The approximate center of the peak.
        :param center_min: The minimum bound for the center of the peak.
        :param center_max: The maximum bound for the center of the peak.
        :param sigma_min: A lower bound for the peak width.
        :param sigma_max: An upper bound for the peak width.
        :param amplitude_min: An lower bound for the amplitude of the peak.
        """
        self.center = peak_center
        self.center_min = center_min
        self.center_max = center_max
        if sigma_min is None:
            self.sigma_min = 0.01
        else:
            self.sigma_min = sigma_min
        if sigma_max is None:
            self.sigma_max = 0.02
        else:
            self.sigma_max = sigma_max
        if amplitude_min is None:
            self.amplitude_min = 0.05
        else:
            self.amplitude_min = amplitude_min


def calc_dspacing(ttheta):
    """ Calculate d-spacing from two-theta values.
    """ 
    x_ray_energy = 89.07  # in keV
    c = 2.99792458e8
    h = 6.62607004e-34
    e = 1.6021766208e-19
    x_ray_wavelength = (h * c) / (x_ray_energy * 1e3 * e)
    
    return x_ray_wavelength / (2 * np.sin(np.array(ttheta) * np.pi / 360))


def calc_strain(ttheta):
    """Calculate strain from two-theta values. Applies average of first 200 points to define
    zero two-theta."""
    theta = 0.5 * (np.array(ttheta)) * np.pi / 180.0
    theta0 = np.mean(theta[0:200])
    strain = -(theta-theta0)/np.tan(theta)
    return strain


def calc_strain_singlepoint(ttheta):
    """Calculate strain from two-theta values. First two-theta values is defined as zero two-theta.
    """
    theta = 0.5 * (np.array(ttheta)) * np.pi / 180.0
    theta0 = theta[0]
    strain = -(theta - theta0) / np.tan(theta)
    return strain


def relative_amplitude(amp):
    """ Calculate difference in amplitude from first measurement."""
    amp0 = amp[2]
    rel_amp = np.array(amp) / amp0
    return rel_amp


# Functions for loading up data and fitting
def get_cake(file_path, cake=1):
    """ Return 'spectrum' containing 2-theta increments and intensity values for a given cake.
        Note, assumed DAWN output data has 2-theta in column 0 and intensity of first cake in
        column 1.
    """
    spectrum = np.loadtxt(file_path, usecols=(0, cake))
    return spectrum
              

def get_spectrum_subset(spectrum, ttheta_lims=(0, 10)):
    """ Return intensity values within a given 2-theta range for an individual lattice plane peak.
        Note, output 'peak' includes 2-theta increments in column 0 and intensity in column 1.
    """
    mask = np.logical_and(spectrum[:, 0] > ttheta_lims[0], spectrum[:, 0] < ttheta_lims[1])
    return spectrum[mask]


def line(x, constBG):
    """constant Background"""
    return constBG


def fit_peak(peak_data, p1: PeakParams):
    """ Pseudo-Voigt fit to the lattice plane peak intensity.
        Return results of the fit as an lmfit class, which contains the fitted parameters (amplitude, fwhm, etc.) 
        and the fit line calculated using the fit parameters and 100x two-theta points.
    """
    ttheta = peak_data[:, 0]
    intensity = peak_data[:, 1]
    pvModel = PseudoVoigtModel()
    model = pvModel + Model(line)

    pars = pvModel.guess(intensity, x=ttheta)
    if p1.center:
        pars['center'].set(p1.center, p1.center_min, p1.center_max)
    pars['sigma'].set(p1.sigma_min, p1.sigma_max)
    pars['amplitude'].set(p1.amplitude_min)
    pars.add("constBG", 0)

    fit_results = model.fit(intensity, pars, x=ttheta)
    fit_ttheta = np.linspace(ttheta[0], ttheta[-1], 100)
    fit_line = [fit_ttheta, model.eval(fit_results.params, x=fit_ttheta)]
    return fit_results, fit_line


def fit_two_peaks(peak_data, peak_1: PeakParams, peak_2: PeakParams):
    ttheta = peak_data[:, 0]
    intensity = peak_data[:, 1]

    PV_1 = PseudoVoigtModel(prefix='pv_1')
    PV_2 = PseudoVoigtModel(prefix='pv_2')

    model = PV_1 + PV_2 + Model(line)

    pars_1 = PV_1.guess(intensity, x=ttheta)
    pars_1['pv_1center'].set(peak_1.center, min=peak_1.center_min, max=peak_1.center_max)
    pars_1['pv_1sigma'].set(min=peak_1.sigma_min, max=peak_1.sigma_max)
    pars_1['pv_1amplitude'].set(min=peak_1.amplitude_min)

    pars_2 = PV_2.guess(intensity, x=ttheta)
    pars_2['pv_2center'].set(peak_2.center, min=peak_2.center_min, max=peak_2.center_max)
    pars_2['pv_2sigma'].set(min=peak_2.sigma_min, max=peak_2.sigma_max)
    pars_2['pv_2amplitude'].set(min=peak_2.amplitude_min)

    pars = pars_1 + pars_2
    pars.add("constBG", 0)

    fit_results = model.fit(intensity, pars, x=ttheta)
    fit_ttheta = np.linspace(ttheta[0], ttheta[-1], 100)
    fit_line = [fit_ttheta, model.eval(fit_results.params, x=fit_ttheta)]
    return fit_results, fit_line


def fit_three_peaks(peak_data, peak_1: PeakParams, peak_2: PeakParams, peak_3: PeakParams):
    ttheta = peak_data[:, 0]
    intensity = peak_data[:, 1]

    PV_1 = PseudoVoigtModel(prefix='pv_1')
    PV_2 = PseudoVoigtModel(prefix='pv_2')
    PV_3 = PseudoVoigtModel(prefix='pv_3')

    model = PV_1 + PV_2 + PV_3 + Model(line)
    
    pars_1 = PV_1.guess(intensity, x=ttheta)
    pars_1['pv_1center'].set(peak_1.center, peak_1.center_min, peak_1.center_max)
    pars_1['pv_1sigma'].set(peak_1.sigma_min, peak_2.sigma_max)
    pars_1['pv_1amplitude'].set(min=peak_1.amplitude_min)

    pars_2 = PV_2.guess(intensity, x=ttheta)
    pars_2['pv_2center'].set(peak_2.center, peak_2.center_min, peak_2.center_max)
    pars_2['pv_2sigma'].set(peak_2.sigma_min, peak_2.sigma_max)
    pars_2['pv_2amplitude'].set(min=peak_2.amplitude_min)

    pars_3 = PV_3.guess(intensity, x=ttheta)
    pars_3['pv_3center'].set(peak_3.center, peak_3.center_min, peak_3.center_max)
    pars_3['pv_3sigma'].set(peak_3.sigma_min, peak_3.sigma_max)
    pars_3['pv_3amplitude'].set(min=peak_3.amplitude_min)

    pars=pars_1 + pars_2 + pars_3
    pars.add("constBG", 0)

    fit_results = model.fit(intensity, pars, x=ttheta)
    fit_ttheta = np.linspace(ttheta[0], ttheta[-1], 100)
    fit_line = [fit_ttheta, model.eval(fit_results.params, x=fit_ttheta)]
    return fit_results, fit_line


class FitPeak(ABC):
    """An object that handles fitting peaks in a spectrum."""
    def __init__(self, file_path, cake):
        self.data_dict = {}
        self.fits_dict = {}
        self.lines_dict = {}
        self.spectrum = get_cake(file_path, cake=cake)
        self.reflection_list = []

    def plot_fit(self, reflection):
        """ Plot the line fit and intensity measurements.
        Input peak labels i.e. (10-10), (0002), etc.
        """
        plt.figure(figsize=(10, 8))
        plt.minorticks_on()
        plt.plot(self.lines_dict[reflection][:, 0], self.lines_dict[reflection][:, 1], linewidth=3)
        plt.plot(self.data_dict[reflection][:, 0], self.data_dict[reflection][:, 1], '+', markersize=15, mew=3)
        plt.xlabel(r'Two Theta ($^\circ$)', fontsize=28)
        # why r? - so can print out latex without needing double slash
        plt.title(reflection, fontsize=28)
        plt.ylabel('Intensity', fontsize=28)
        plt.tight_layout()

    def plot_spectrum(self, xmin=0, xmax=10):
        """Plot the intensity spectrum."""
        plt.figure(figsize=(10, 8))
        plt.minorticks_on()
        plt.plot(self.spectrum[:, 0], self.spectrum[:, 1], '-', linewidth=3)
        plt.xlabel(r'Two Theta ($^\circ$)', fontsize=28)
        plt.ylabel('Intensity', fontsize=28)
        plt.xlim(xmin, xmax)
        plt.tight_layout()


class FitSingletPeak(FitPeak):
    """ Class for reading in individual cakes and fitting multiple single peaks
        See examples below for usage.
    """

    def __init__(self, file_path, cake):
        super().__init__(file_path, cake)

    def fit_peaks(self, reflection_list: List[str], peak_ranges: List[tuple], peak_params: List[PeakParams]):
        """ Attempt to fit peaks within the ranges specified by `peak_ranges`.
        :param reflection_list: One label for each peak.
        :param peak_ranges: A tuple for each peak specifying where on the x-axis the peak begins
        and ends.
        """
        self.reflection_list = reflection_list
        # zip iterates through each list together
        for reflection, p_range, p1 in zip(reflection_list, peak_ranges, peak_params):
            # store data in dictionary with peak label as the key
            self.data_dict[reflection] = get_spectrum_subset(self.spectrum, ttheta_lims=p_range)
            fit_results, fit_line = fit_peak(self.data_dict[reflection], p1)
            self.fits_dict[reflection] = fit_results
            # Transpose the array to get appropriate row/column order.
            self.lines_dict[reflection] = np.array(fit_line).T


class FitDoubletPeak(FitPeak):
    """ Class for reading in individual cakes and fitting two peaks at the same time
    """
    def __init__(self, file_path, cake):
        super().__init__(file_path, cake)

    def fit_2_peaks(self, reflection_list, peak_ranges, p1: PeakParams, p2: PeakParams):
        self.reflection_list = reflection_list
        for reflection, p_range in zip(reflection_list, peak_ranges):
            peak_data = get_spectrum_subset(self.spectrum, ttheta_lims=p_range)
            self.data_dict[reflection] = peak_data
        for reflection, peak_data in self.data_dict.items():
            fit_results, fit_line = fit_two_peaks(peak_data, p1, p2)
            self.fits_dict[reflection] = fit_results
            self.lines_dict[reflection] = np.array(fit_line).T


class FitTripletPeak(FitPeak):
    """ Class for reading in individual cakes and fitting three peaks at the same time
    """
    def __init__(self, file_path, cake):
        super().__init__(file_path, cake)

    def fit_3_peaks(self, reflection_list, peak_ranges, p1: PeakParams, p2: PeakParams,
                    p3: PeakParams):
        self.reflection_list = reflection_list
        # zip iterates through each list together
        for reflection, p_range in zip(reflection_list, peak_ranges):
            peak_data = get_spectrum_subset(self.spectrum, ttheta_lims=p_range)
            self.data_dict[reflection] = peak_data
            # store data in dictionary with peak label as the key
        for reflection, peak_data in self.data_dict.items():
            fit_results, fit_line = fit_three_peaks(peak_data, p1, p2, p3, initParams=None)
            self.fits_dict[reflection] = fit_results
            self.lines_dict[reflection] = np.array(fit_line).T
