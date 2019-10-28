import glob
from typing import List, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lmfit
from tqdm import tqdm
import pandas as pd

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)


class MaximumParams:
    """An object containing fitting details of a single maximum within a peak."""
    def __init__(self, peak_center: float = None, center_min: float = None,
                 center_max: float = None, sigma_min: float = None,
                 sigma_max: float = None, amplitude_min: float = None):
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
    :ivar result: The lmfit result of the fit.
    :ivar cake_numbers: The cake number each column in raw_spectrum refers to."""
    def __init__(self, name: str):
        self.name = name
        self.raw_spectrum: Union[None, np.ndarray] = None
        self.result: Union[None, lmfit.model.ModelResult] = None
        self.cake_numbers: List[int] = []

    def plot(self, num_fit_points=100):
        """ Plot the raw spectral data and the fit."""
        if self.raw_spectrum is None:
            print("Cannot plot fit peak as fitting has not been done yet.")
        else:
            plt.figure(figsize=(8, 6))
            plt.minorticks_on()
            plt.tight_layout()
            plt.xlabel(r'Two Theta ($^\circ$)')
            plt.ylabel('Intensity')
            for index, cake_num in enumerate(self.cake_numbers):
                plt.plot(self.raw_spectrum[:, 0], self.raw_spectrum[:, index + 1], '+', ms=15,
                         mew=3, label="Cake {}".format(cake_num))
            x_data = np.linspace(np.min(self.raw_spectrum[:, 0]), np.max(self.raw_spectrum[:, 0]), num_fit_points)
            y_fit = self.result.model.eval(self.result.params, x=x_data)
            plt.plot(x_data, y_fit, 'k--', lw=1, label="Fit")
            plt.legend()
            plt.title(self.name)
            plt.tight_layout()
            plt.show()


def do_pv_fit(peak_data: np.ndarray, peak_params: List[MaximumParams]):
    """
    Pseudo-Voigt fit to the lattice plane peak intensity.
    Return results of the fit as an lmfit class, which contains the fitted parameters
    (amplitude, fwhm, etc.) and the fit line calculated using the fit parameters and
    100x two-theta points.
    """
    two_theta = peak_data[:, 0]
    intensity = peak_data[:, 1]

    combined_model = None
    combined_parameters = None

    for index, peak in enumerate(peak_params):
        # Add the peak to the model
        peak_prefix = "peak_{}_".format(index + 1)
        model = lmfit.models.PseudoVoigtModel(prefix=peak_prefix)
        if combined_model:
            combined_model += model
        else:
            combined_model = model

        # Add the fit parameters for the peak
        parameters = model.guess(intensity, x=two_theta)
        if peak.center:
            parameters['{}center'.format(peak_prefix)].set(peak.center, min=peak.center_min,
                                                           max=peak.center_max)
        parameters['{}sigma'.format(peak_prefix)].set(min=peak.sigma_min, max=peak.sigma_max)
        parameters['{}amplitude'.format(peak_prefix)].set(min=peak.amplitude_min)
        if combined_parameters:
            combined_parameters += parameters
        else:
            combined_parameters = parameters
    combined_model += lmfit.Model(lambda constant_background: constant_background)
    combined_parameters.add("constant_background", 0)

    fit_results = combined_model.fit(intensity, combined_parameters, x=two_theta)
    return fit_results


class FitSpectrum:
    """An object that handles fitting peaks in a spectrum.
    :ivar verbose: Whether or not to print fit status to the console.
    :ivar first_cake_angle: The angle of the first cake in the data file in degrees
    clockwise from North.
    :ivar spectral_data: Data for the whole diffraction pattern.
    :ivar fitted_peaks: Fits to peaks in the spectrum.
    """
    def __init__(self, file_path: str, first_cake_angle: int = 90, verbose: bool = True):
        self.verbose = verbose
        self.first_cake_angle = first_cake_angle
        self.fitted_peaks: List[PeakFit] = []

        self.spectral_data = pd.read_table(file_path).to_numpy()
        if self.verbose:
            print("Diffraction pattern successfully loaded from file.")

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
        plt.minorticks_on()
        plt.xlabel(r'Two Theta ($^\circ$)')
        plt.ylabel('Intensity')
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
            if merge_cakes:
                new_fit.cake_numbers = [" + ".join(map(str, cakes))]
                new_fit.result = do_pv_fit(new_fit.raw_spectrum, peak.maxima)
            else:
                new_fit.cake_numbers = list(map(str, cakes))
                stacked_spectrum = get_stacked_spectrum(new_fit.raw_spectrum)
                new_fit.result = do_pv_fit(stacked_spectrum, peak.maxima)
            self.fitted_peaks.append(new_fit)
        if self.verbose:
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

    def get_fit(self, name: str) -> Union[PeakFit, None]:
        """Get a peak fit by name."""
        for fit in self.fitted_peaks:
            if fit.name == name:
                return fit
        return None


class FittingExperiment:
    """Information about a series of fits to temporally spaced diffraction patterns.
    :ivar frame_time: Time between subsequent diffraction patterns.
    :ivar file_stub: String used to glob for the diffraction patterns.
    :ivar first_cake_angle:"""
    def __init__(self, frame_time: int, file_string: str, first_cake_angle: int,
                 cakes_to_fit: List[int], peak_params: Union[PeakParams, List[PeakParams]],
                 merge_cakes: bool, frames_to_load: List[int] = None):
        self.frame_time = frame_time
        self.file_string = file_string
        self.first_cake_angle = first_cake_angle
        self.cakes_to_fit = cakes_to_fit
        if isinstance(peak_params, PeakParams):
            peak_params = [peak_params]
        self.peak_params = peak_params
        self.merge_cakes = merge_cakes
        self.frames_to_load = frames_to_load

        self.spectra_fits = []

    def run_analysis(self):
        """Iterate a fit over multiple diffusion patterns."""
        if self.frames_to_load:
            file_list = [self.file_string.format(number) for number in self.frames_to_load]
        else:
            file_list = sorted(glob.glob(self.file_string))

        print("Processing {} diffusion patterns.".format(len(file_list)))
        iteration_peak_params = self.peak_params
        for file_path in tqdm(file_list):
            spectral_data = FitSpectrum(file_path, self.first_cake_angle, verbose=False)
            spectral_data.fit_peaks(self.cakes_to_fit, iteration_peak_params, self.merge_cakes)
            self.spectra_fits.append(spectral_data)
            # Here is where a function would go to pass the old fit onto the new fit.
            iteration_peak_params = self.peak_params
        print("Analysis complete.")

    def list_fits(self) -> List[str]:
        """List the peaks that have been fitted."""
        return [peak.name for peak in self.peak_params]

    def plot_fit_parameter(self, peak_name: str, fit_parameter: str):
        """Plot a named parameter of a fit as a function of time.
        :param peak_name: The name of the fit to plot.
        :param fit_parameter: The name of the fit parameter to plot.
        """
        if peak_name in [peak.name for peak in self.peak_params]:
            peak_heights = []
            for timestep in self.spectra_fits:
                peak_fit = timestep.get_fit(peak_name)
                peak_heights.append(
                    peak_fit.result.params["peak_{}_{}".format(peak_name, fit_parameter)])
            plt.plot((np.arange(len(peak_heights)) + 1) * self.frame_time, peak_heights)
            plt.xlabel("Time (s)")
            plt.ylabel("Peak {}".format(fit_parameter))
            plt.show()
        else:
            print("Peak '{}' not found in fitted peaks.")


def get_stacked_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Take an number of observations from N different cakes and stack them vertically into a 2
    column wide array"""
    stacked_data = spectrum[:, 0:2]
    spectrum_columns = spectrum.shape[1]
    for column_num in range(2, spectrum_columns):
        stacked_data = np.vstack(
            (stacked_data, spectrum[np.ix_([True] * spectrum.shape[0], [0, column_num])]))
    stacked_data = stacked_data[stacked_data[:, 0].argsort()]
    return stacked_data
