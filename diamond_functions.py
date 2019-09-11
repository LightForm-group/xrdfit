import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import PseudoVoigtModel
from lmfit import Model
from typing import List


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


def do_pv_fit(peak_data, peak_params: List[PeakParams]):
    """
    Pseudo-Voigt fit to the lattice plane peak intensity.
    Return results of the fit as an lmfit class, which contains the fitted parameters
    (amplitude, fwhm, etc.) and the fit line calculated using the fit parameters and
    100x two-theta points.
    """
    ttheta = peak_data[:, 0]
    intensity = peak_data[:, 1]

    combined_model = None
    combined_parameters = None

    for index, peak in enumerate(peak_params):
        # Add the peak to the model
        peak_prefix = "peak_{}_".format(index + 1)
        model = PseudoVoigtModel(prefix=peak_prefix)
        if combined_model:
            combined_model += model
        else:
            combined_model = model

        # Add the fit parameters for the peak
        parameters = model.guess(intensity, x=ttheta)
        if peak_params[index].center:
            parameters['{}center'.format(peak_prefix)].set(peak_params[index].center, min=peak_params[index].center_min, max=peak_params[index].center_max)
        parameters['{}sigma'.format(peak_prefix)].set(min=peak_params[index].sigma_min, max=peak_params[index].sigma_max)
        parameters['{}amplitude'.format(peak_prefix)].set(min=peak_params[index].amplitude_min)
        if combined_parameters:
            combined_parameters += parameters
        else:
            combined_parameters = parameters
    combined_model += Model(line)
    combined_parameters.add("constBG", 0)

    fit_results = combined_model.fit(intensity, combined_parameters, x=ttheta)
    fit_ttheta = np.linspace(ttheta[0], ttheta[-1], 100)
    fit_line = [fit_ttheta, combined_model.eval(fit_results.params, x=fit_ttheta)]
    return fit_results, fit_line


class FitPeak():
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

    def fit_peaks(self, reflection_list: List[str], peak_ranges: List[tuple], peak_params: List[List[PeakParams]]):
        """ Attempt to fit peaks within the ranges specified by `peak_ranges`.
        :param reflection_list: One label for each peak.
        :param peak_ranges: A tuple for each peak specifying where on the x-axis the peak begins
        and ends.
        :param peak_params: One list of PeakParams for each peak in the spectrum.
        """
        self.reflection_list = reflection_list
        for reflection, p_range, p1 in zip(reflection_list, peak_ranges, peak_params):
            self.data_dict[reflection] = get_spectrum_subset(self.spectrum, ttheta_lims=p_range)
            fit_results, fit_line = do_pv_fit(self.data_dict[reflection], p1)
            self.fits_dict[reflection] = fit_results
            # Transpose the array to get appropriate row/column order.
            self.lines_dict[reflection] = np.array(fit_line).T