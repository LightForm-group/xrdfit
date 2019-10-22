from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import PseudoVoigtModel
from lmfit import Model


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
    def __init__(self, name: str, peak_range: Tuple[float, float],
                 maxima: List[MaximumParams] = None):
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
            plt.plot(self.raw_spectrum[:, 0], self.raw_spectrum[:, 1], 'b+', ms=15, mew=3,
                     label="Spectrum")
            plt.plot(self.points[:, 0], self.points[:, 1], 'k--', lw=1, label="Fit")
            plt.legend()
            plt.title(self.name, fontsize=label_size)
            plt.tight_layout()


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
            parameters['{}center'.format(peak_prefix)].set(peak_params[index].center,
                                                           min=peak_params[index].center_min,
                                                           max=peak_params[index].center_max)
        parameters['{}sigma'.format(peak_prefix)].set(min=peak_params[index].sigma_min,
                                                      max=peak_params[index].sigma_max)
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
    """An object that handles fitting peaks in a spectrum.
    :ivar spectral_data: (NumPy array) A numpy array containing the whole diffraction pattern.
    :ivar fitted_peaks: List[Lmfit result] Fits to the peaks in the spectrum.
    """
    def __init__(self, file_path: str, first_cake_angle: int = 90):
        self.spectral_data = np.loadtxt(file_path)
        print("Diffraction pattern successfully loaded from file.")
        self.fitted_peaks = []
        self.first_cake_angle = first_cake_angle

    def plot_polar(self):
        """Plot the whole diffraction pattern on polar axes."""
        with np.errstate(divide='ignore'):
            z_data = np.log10(self.spectral_data[:, 1:])
        rad = self.spectral_data[:, 0]
        num_cakes = z_data.shape[1]
        self._plot_polar_heatmap(num_cakes, rad, z_data)

    def highlight_cakes(self, cakes: Union[int, List[int]]):
        """Plot a circular map of the cakes with the selected cakes highlighted."""
        num_cakes = self.spectral_data.shape[1] - 1
        z_data = np.zeros((1, self.spectral_data.shape[1] - 1))
        for cake_num in cakes:
            z_data[0, cake_num - 1] = 1
        rad = [0, 1]
        self._plot_polar_heatmap(num_cakes, rad, z_data)

    def _plot_polar_heatmap(self, num_cakes, rad, z_data):
        """A method for plotting a polar heatmap."""
        azm = np.linspace(0, 2 * np.pi, num_cakes + 1)
        r, theta = np.meshgrid(rad, azm)
        plt.subplot(projection="polar", theta_direction=-1,
                    theta_offset=np.deg2rad(360 / num_cakes / 2))
        plt.pcolormesh(theta, r, z_data.T)
        plt.plot(azm, r, ls='none')
        plt.grid()
        # Turn on theta grid lines at the cake edges
        plt.thetagrids([theta * 360 / num_cakes for theta in range(num_cakes)], labels=[])
        # Turn off radial grid lines
        plt.rgrids([])
        # Put the cake numbers in the right places
        ax = plt.gca()
        trans, _, _ = ax.get_xaxis_text1_transform(0)
        for label in range(1, num_cakes + 1):
            ax.text(np.deg2rad(label * 10 - 95 + self.first_cake_angle), -0.1, label,
                    transform=trans, rotation=0, ha="center", va="center")
        plt.show()

    def plot(self, cakes_to_plot: Union[int, List[int]], x_min: float = 0, x_max: float = 10,
             merge_cakes: bool = False):
        """Plot the intensity as a function of two theta for a given cake."""
        if isinstance(cakes_to_plot, int):
            cakes_to_plot = [cakes_to_plot]

        # Plot the data
        plt.figure(figsize=(8, 6))
        if merge_cakes:
            data = self.get_spectrum_subset(cakes_to_plot, (x_min, x_max), True)
            plt.plot(data[:, 0], data[:, 1:], '-', linewidth=2)
        else:
            for cake_num in cakes_to_plot:
                plt.plot(self.spectral_data[:, 0], self.spectral_data[:, cake_num], '-',
                         linewidth=2, label=cake_num)
                plt.legend()

        # Plot formatting
        label_size = 20
        plt.minorticks_on()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'Two Theta ($^\circ$)', fontsize=label_size)
        plt.ylabel('Intensity', fontsize=label_size)
        plt.xlim(x_min, x_max)
        plt.tight_layout()
        plt.show()

    def fit_peaks(self, cakes: Union[int, List[int]],
                  peak_params: Union[PeakParams, List[PeakParams]], merge_cakes: bool = False):
        """Attempt to fit peaks within the ranges specified by `peak_ranges`.
        :param peak_params: A list of PeakParams describing the peaks to be fitted.
        :param cakes: Which cakes to fit.
        :param merge_cakes: If True and multiple cakes are specified then sum the cakes before
        fitting. Else do the fit to multiple cakes simultaneously.
        """
        self.fitted_peaks = []
        if isinstance(cakes, int):
            cakes = [cakes]
        if isinstance(peak_params, PeakParams):
            peak_params = [peak_params]

        for peak in peak_params:
            new_fit = PeakFit(peak.name)
            new_fit.raw_spectrum = self.get_spectrum_subset(cakes, peak.range, merge_cakes)
            fit_results, fit_line = do_pv_fit(new_fit.raw_spectrum, peak.maxima)
            new_fit.result = fit_results
            # Transpose the array to get appropriate row/column order.
            new_fit.points = np.array(fit_line).T
            self.fitted_peaks.append(new_fit)
        print("Fitting complete.")

    def get_spectrum_subset(self, cakes: Union[int, List[int]],
                            two_theta_lims: Tuple[float, float],
                            merge_cakes: bool) -> np.ndarray:
        """Return spectral intensity as a function of 2-theta for a selected 2-theta range.
        :param cakes: One or more cakes to get the intensity for.
        :param two_theta_lims: Limits to the two-theta values returned.
        :param merge_cakes: If more than one cake and True, sum the values of all of the cakes
        else return one column for each cake."""
        if isinstance(cakes, int):
            cakes = [cakes]

        theta_mask = np.logical_and(self.spectral_data[:, 0] > two_theta_lims[0],
                                    self.spectral_data[:, 0] < two_theta_lims[1])
        if merge_cakes:
            # Returns 2 columns, the two-theta angles and the summed intensity
            data = np.sum(self.spectral_data[:, cakes], axis=1)
            data = np.vstack((self.spectral_data[:, 0], data)).T
            return data[theta_mask, :]
        else:
            # Returns an array with one column for the two-theta values and one column for each cake
            chosen_cakes = [0] + cakes
            return self.spectral_data[np.ix_(theta_mask, chosen_cakes)]

    def get_fit(self, name: str):
        """Get a peak fit by name."""
        for fit in self.fitted_peaks:
            if fit.name == name:
                return fit
        return None
