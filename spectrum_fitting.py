import bz2
import glob
from typing import List, Tuple, Union

import numpy as np

import lmfit
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import dill

import plotting


class PeakParams:
    """An object containing information about a peak and its maxima.
    :ivar name: The name of the peak.
    :ivar range: Where in the spectrum the peak begins and ends.
    :ivar num_maxima: The number of maxima in this peak.
    :ivar maxima_ranges: If there is more than one maxima, where each maximum begins and ends."""
    def __init__(self, name: str, peak_range: Tuple[float, float], num_maxima: int = 1,
                 maxima_ranges: dict = None):
        self.name = name
        self.range = peak_range
        self.num_maxima = num_maxima
        if maxima_ranges:
            self.maxima_ranges = maxima_ranges
        else:
            self.maxima_ranges = {"1_min": peak_range[0], "1_max": peak_range[1]}

        self._check_maxima_ranges()
        self.previous_fit: Union[lmfit.Parameters, None] = None

    def set_previous_fit(self, fit_parameters: lmfit.Parameters):
        """Since the peak center may drift over time the limits must be reset otherwise it
         will always have the limits from the first fit."""
        for parameter in fit_parameters:
            if "center" in parameter:
                center_value = fit_parameters[parameter]
                center_min = center_value - 0.02
                center_max = center_value + 0.02
                fit_parameters[parameter].set(value=None, min=center_min, max=center_max)
        self.previous_fit = fit_parameters

    def _check_maxima_ranges(self):
        """Check that the user has provided the correct number of maxima ranges with the right
        names for the number of peaks to be fitted."""
        for peak_num in range(1, self.num_maxima + 1):
            for min_max in ["min", "max"]:
                param = f"{peak_num}_{min_max}"
                if param not in self.maxima_ranges:
                    raise TypeError(f"Maxima descriptor {param} is missing.")


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

    def plot(self):
        """ Plot the raw spectral data and the fit."""
        if self.raw_spectrum is None:
            print("Cannot plot fit peak as fitting has not been done yet.")
        else:
            plotting.plot_peak_fit(self.raw_spectrum, self.cake_numbers, self.result, self.name)


def do_pv_fit(peak_data: np.ndarray, num_maxima: int, maxima_ranges: dict,
              fit_parameters: lmfit.Parameters = None):
    """
    Pseudo-Voigt fit to the lattice plane peak intensity.
    Return results of the fit as an lmfit class, which contains the fitted parameters
    (amplitude, fwhm, etc.) and the fit line calculated using the fit parameters and
    100x two-theta points.
    """
    model = None
    peak_prefix = "maximum_{}_"

    for maxima_num in range(num_maxima):
        # Add the peak to the model
        if model:
            model += lmfit.models.PseudoVoigtModel(prefix=peak_prefix.format(maxima_num + 1))
        else:
            model = lmfit.models.PseudoVoigtModel(prefix=peak_prefix.format(maxima_num + 1))
    model += lmfit.Model(lambda background: background)

    two_theta = peak_data[:, 0]
    intensity = peak_data[:, 1]

    if not fit_parameters:
        fit_parameters = guess_params(two_theta, intensity, maxima_ranges)

    fit_result = model.fit(intensity, fit_parameters, x=two_theta,
                           fit_kws={"xtol": 1e-7}, iter_cb=iteration_callback)

    return fit_result


def guess_params(x_data, y_data, maxima_ranges: dict) -> lmfit.Parameters:
    """Given a dataset and some details about where the maxima are, guess some good initial
    values for the PV fit."""
    params = lmfit.Parameters()
    num_maxima = len(maxima_ranges) // 2
    for index in range(1, num_maxima + 1):
        maximum_mask = np.logical_and(x_data < maxima_ranges[f"{index}_max"],
                                      x_data > maxima_ranges[f"{index}_min"])
        maxima_x = x_data[maximum_mask]
        maxima_y = y_data[maximum_mask]
        center = maxima_x[np.argmax(maxima_y)]
        sigma = 0.02
        amplitude = ((max(maxima_y) - min(maxima_y)) * 3) * sigma

        params.add(f"maximum_{index}_center", value=center, min=center - 0.02, max=center + 0.02)
        params.add(f"maximum_{index}_sigma", value=sigma, min=0.01, max=0.02)
        params.add(f"maximum_{index}_fraction", value=0.2, min=0, max=1)
        params.add(f"maximum_{index}_amplitude", value=amplitude, min=0.05)
    params.add("background", value=min(y_data), min=0, max=max(y_data))
    return params


def iteration_callback(params, iter, resid, *args, **kws):
    """This method is called on every iteration of the minimisation. This can be used
    to monitor progress."""
    return False


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
        plotting.plot_polar_heatmap(num_cakes, rad, z_data, self.first_cake_angle)

    def highlight_cakes(self, cakes: Union[int, List[int]]):
        """Plot a circular map of the cakes with the selected cakes highlighted."""
        num_cakes = self.spectral_data.shape[1] - 1
        z_data = np.zeros((1, self.spectral_data.shape[1] - 1))
        for cake_num in cakes:
            z_data[0, cake_num - 1] = 1
        rad = [0, 1]
        plotting.plot_polar_heatmap(num_cakes, rad, z_data, self.first_cake_angle)

    def plot(self, cakes_to_plot: Union[int, List[int]], x_range: Tuple[float, float] = (0, 10),
             merge_cakes: bool = False, show_points=False):
        """Plot the intensity as a function of two theta for a given cake."""
        if isinstance(cakes_to_plot, int):
            cakes_to_plot = [cakes_to_plot]
        # Get the data to plot
        if merge_cakes:
            data = self.get_spectrum_subset(cakes_to_plot, x_range, True)
        else:
            data = self.spectral_data
        plotting.plot_spectrum(data, cakes_to_plot, merge_cakes, show_points, x_range)
        plt.show()

    def plot_peak_params(self, peak_params: List[PeakParams], cakes_to_plot: Union[int, List[int]],
                         x_range: Tuple[float, float] = (0, 10), merge_cakes: bool = False,
                         show_points=False):
        if isinstance(cakes_to_plot, int):
            cakes_to_plot = [cakes_to_plot]
        # Get the data to plot
        if merge_cakes:
            data = self.get_spectrum_subset(cakes_to_plot, x_range, True)
        else:
            data = self.spectral_data
        plotting.plot_spectrum(data, cakes_to_plot, merge_cakes, show_points, x_range)
        plotting.plot_peak_params(peak_params)
        plt.show()

    def fit_peaks(self, peak_params: Union[PeakParams, List[PeakParams]],
                  cakes: Union[int, List[int]], merge_cakes: bool = False):
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
                new_fit.result = do_pv_fit(new_fit.raw_spectrum, peak.num_maxima,
                                           peak.maxima_ranges, peak.previous_fit)
            else:
                new_fit.cake_numbers = list(map(str, cakes))
                stacked_spectrum = get_stacked_spectrum(new_fit.raw_spectrum)
                new_fit.result = do_pv_fit(stacked_spectrum, peak.num_maxima, peak.maxima_ranges,
                                           peak.previous_fit)
            self.fitted_peaks.append(new_fit)
        if self.verbose:
            print("Fitting complete.")

    def get_spectrum_subset(self, cakes: Union[int, List[int]], two_theta_lims: Tuple[float, float],
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

    def get_fit(self, name: str) -> Union[PeakFit]:
        """Get a peak fit by name."""
        for fit in self.fitted_peaks:
            if fit.name == name:
                return fit
        raise KeyError(f"Fit: '{name}' not found")


class FittingExperiment:
    """Information about a series of fits to temporally spaced diffraction patterns.
    :ivar spectrum_time: Time between subsequent diffraction patterns.
    :ivar file_string: String used to glob for the diffraction patterns.
    :ivar first_cake_angle:"""
    def __init__(self, spectrum_time: int, file_string: str, first_cake_angle: int,
                 cakes_to_fit: List[int], peak_params: Union[PeakParams, List[PeakParams]],
                 merge_cakes: bool, frames_to_load: List[int] = None, reuse_fits=False):
        self.spectrum_time = spectrum_time
        self.file_string = file_string
        self.first_cake_angle = first_cake_angle
        self.cakes_to_fit = cakes_to_fit
        if isinstance(peak_params, PeakParams):
            peak_params = [peak_params]
        self.peak_params = peak_params
        self.merge_cakes = merge_cakes
        self.frames_to_load = frames_to_load
        self.reuse_fits = reuse_fits

        self.timesteps: List[FitSpectrum] = []

    def run_analysis(self):
        """Iterate a fit over multiple diffraction patterns."""
        if self.frames_to_load:
            file_list = [self.file_string.format(number) for number in self.frames_to_load]
        else:
            file_list = sorted(glob.glob(self.file_string))

        print("Processing {} diffraction patterns.".format(len(file_list)))
        for file_path in tqdm(file_list):
            spectral_data = FitSpectrum(file_path, self.first_cake_angle, verbose=False)
            spectral_data.fit_peaks(self.peak_params, self.cakes_to_fit, self.merge_cakes)
            self.timesteps.append(spectral_data)

            if self.reuse_fits:
                # Pass the results of the fit on to the next time step.
                for peak_fit, peak_params in zip(spectral_data.fitted_peaks, self.peak_params):
                    peak_params.set_previous_fit(peak_fit.result.params)

        print("Analysis complete.")

    def peak_names(self) -> List[str]:
        """List the peaks that have been fitted."""
        return [peak.name for peak in self.peak_params]

    def fit_parameters(self, peak_name) -> List[str]:
        """List the parameters of the fit for a specified peak.
        :param peak_name: The peak to list the parameters of.
        """
        return self.timesteps[0].get_fit(peak_name).result.var_names

    def get_fit_parameter(self, peak_name: str,
                          fit_parameter: str) -> Union[None, np.ndarray]:
        """Get a fitting parameter over time."""
        if peak_name in [peak.name for peak in self.peak_params]:
            if fit_parameter in self.fit_parameters(peak_name):
                parameters = []
                for timestep in self.timesteps:
                    peak_fit = timestep.get_fit(peak_name)
                    parameters.append(peak_fit.result.params[fit_parameter])
                if self.frames_to_load:
                    x_data = np.array(self.frames_to_load) * self.spectrum_time
                else:
                    x_data = (np.arange(len(parameters)) + 1) * self.spectrum_time
                data = np.vstack((x_data, parameters)).T
                return data
            else:
                print("Unknown fit parameter {} for peak {}".format(fit_parameter, peak_name))
                return None
        else:
            print("Peak '{}' not found in fitted peaks.")
            return None

    def plot_fit_parameter(self, peak_name: str, fit_parameter: str, show_points=False):
        """Plot a named parameter of a fit as a function of time.
        :param peak_name: The name of the fit to plot.
        :param fit_parameter: The name of the fit parameter to plot.
        :param show_points: Whether to show data points on the plot.
        """
        data = self.get_fit_parameter(peak_name, fit_parameter)
        if data is not None:
            plotting.plot_parameter(data, fit_parameter, peak_name, show_points)

    def save(self, file_name: str):
        """Dump the object to a compressed binary file using dill."""
        print("Saving data to dump file.")
        with bz2.open(file_name, 'wb') as output_file:
            dill.dump(self, output_file)
        print("Data successfully saved to dump file.")


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


def load_dump(file_name: str) -> FittingExperiment:
    """Load a FittingExperiment object saved using the FittingExperiment.save method."""
    print("Loading data from dump file.")
    with bz2.open(file_name, "rb") as input_file:
        data = dill.load(input_file)
        print("Data successfully loaded from dump file.")
        return data
