import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import PseudoVoigtModel
from lmfit import Model
from typing import List, Tuple

import averaging_angles


class MaximumParams:
    """An object containing fitting details of a single maximum within a peak."""
    def __init__(self, peak_center=None, center_min=None, center_max=None, sigma_min=None,
                 sigma_max=None, amplitude_min=None):
        """
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


class PeakParams:
    """An object containing information about a peak and its maxima."""
    def __init__(self, name: str, peak_range: Tuple[int, int], maxima: List[MaximumParams] = None):
        self.name = name
        self.range = peak_range
        if maxima:
            self.maxima = maxima
        else:
            self.maxima = [MaximumParams()]


class PeakFit:
    """An object containing data on the fit to a peak.
    :ivar name: The name of the peak.
    :ivar raw_spectrum: The raw data to which the fit is made.
    :ivar points: The fit evaluated over the range of raw_spectrum.
    :ivar result: The lmfit result of the fit."""
    def __init__(self, name: str):
        self.name = name
        self.raw_spectrum = None
        self.points = None
        self.result = None

    def plot(self):
        """ Plot the raw spectral data and the fit."""
        if self.raw_spectrum is None:
            print("Cannot plot fit peak as fitting has not been done yet.")
        else:
            plt.figure(figsize=(8, 6))
            label_size = 20
            plt.minorticks_on()
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.xlabel(r'Two Theta ($^\circ$)', fontsize=label_size)
            plt.ylabel('Intensity', fontsize=label_size)
            plt.plot(self.raw_spectrum[:, 0], self.raw_spectrum[:, 1], 'b+', ms=15, mew=3, label="Spectrum")
            plt.plot(self.points[:, 0], self.points[:, 1], 'k--', lw=1, label="Fit")
            plt.legend()
            plt.title(self.name, fontsize=label_size)
            plt.tight_layout()


def get_spectrum_subset(spectrum, ttheta_lims=(0, 10)):
    """ Return intensity values within a given 2-theta range for an individual lattice plane peak.
        Note, output 'peak' includes 2-theta increments in column 0 and intensity in column 1.
    """
    mask = np.logical_and(spectrum[:, 0] > ttheta_lims[0], spectrum[:, 0] < ttheta_lims[1])
    return spectrum[mask]


def line(x, constBG):
    """constant Background"""
    return constBG


def do_pv_fit(peak_data, peak_params: List[MaximumParams]):
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
    fit_x_data = np.linspace(ttheta[0], ttheta[-1], 100)
    fit_line = [fit_x_data, combined_model.eval(fit_results.params, x=fit_x_data)]
    return fit_results, fit_line


class FitSpectrum:
    """An object that handles fitting peaks in a spectrum."""
    def __init__(self):
        self.spectrum = None
        self.fitted_peaks = []

    def load_merged_spectrum(self, file_path: str, starting_angle: int, averaging_type: str):
        data = np.loadtxt(file_path)
        num_cakes = data.shape[1] - 1

        if averaging_type not in averaging_angles.ANGLES:
            print("Data not loaded. {} is an unknown averaging type. "
                  "Use one of: {}".format(averaging_type, averaging_angles.ANGLES.keys()))
        else:
            cakes_to_average = averaging_angles.get_cakes_to_average(averaging_type, num_cakes,
                                                                     starting_angle)
            spectral_data = np.mean(data[:, cakes_to_average], axis=1)
            self.spectrum = np.vstack((data[:, 0], spectral_data)).T
            print("Spectrum successfully loaded from file.")

    def load_single_spectrum(self, file_path, cake):
        self.spectrum = np.loadtxt(file_path, usecols=(0, cake))
        print("Spectrum successfully loaded from file.")

    def plot(self, x_min=0, x_max=10):
        """Plot the intensity spectrum."""
        plt.figure(figsize=(8, 6))
        label_size = 20
        plt.minorticks_on()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.plot(self.spectrum[:, 0], self.spectrum[:, 1], '-', linewidth=3)
        plt.xlabel(r'Two Theta ($^\circ$)', fontsize=label_size)
        plt.ylabel('Intensity', fontsize=label_size)
        plt.xlim(x_min, x_max)
        plt.tight_layout()

    def fit_peaks(self, peak_params: List[PeakParams]):
        """Attempt to fit peaks within the ranges specified by `peak_ranges`.
        :param peak_params: A list of PeakParams describing the peaks to be fitted.
        """
        self.fitted_peaks = []
        if isinstance(peak_params, PeakParams):
            peak_params = [peak_params]
        for peak in peak_params:
            new_fit = PeakFit(peak.name)
            new_fit.raw_spectrum = get_spectrum_subset(self.spectrum, ttheta_lims=peak.range)
            fit_results, fit_line = do_pv_fit(new_fit.raw_spectrum, peak.maxima)
            new_fit.result = fit_results
            # Transpose the array to get appropriate row/column order.
            new_fit.points = np.array(fit_line).T
            self.fitted_peaks.append(new_fit)

    def get_fit(self, name: str):
        """Get a peak fit by name."""
        for fit in self.fitted_peaks:
            if fit.name == name:
                return fit
        return None
