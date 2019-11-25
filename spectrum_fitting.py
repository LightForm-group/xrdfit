import bz2
import glob
from typing import List, Tuple, Union

import numpy as np
from scipy.signal import find_peaks
import lmfit
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import dill

import plotting
from pv_fit import do_pv_fit


class PeakParams:
    """An object containing information about a peak and its maxima.
    :ivar name: A name for the peak.
    :ivar peak_bounds: Where in the spectrum the peak begins and ends. The fit will be done over
    this region.
    :ivar maxima_bounds: If there is more than one maxima, a bounding box for each peak center.
    """
    def __init__(self, name: str, peak_bounds: Tuple[float, float],
                 maxima_bounds: List[Tuple[float, float]] = None):
        self.name = name
        self.peak_bounds = peak_bounds
        if maxima_bounds:
            self.maxima_bounds = maxima_bounds
            self._check_maxima_bounds()
        else:
            self.maxima_bounds = [peak_bounds]

        self.previous_fit_parameters: Union[lmfit.Parameters, None] = None

    def set_previous_fit(self, previous_fit_parameters: lmfit.Parameters):
        """When passing the result of a previous fit to the next fit, the peak center may drift
        over time. The limits for the next fit must be reset otherwise it will always have the
        limits from the first fit."""
        for parameter in previous_fit_parameters.values():
            if "center" in parameter.name:
                center_value = parameter.value
                center_range = parameter.max - parameter.min
                center_min = center_value - (center_range / 2)
                center_max = center_value + (center_range / 2)
                previous_fit_parameters.add(parameter.name, value=center_value, min=center_min,
                                            max=center_max)
        self.previous_fit_parameters = previous_fit_parameters

    def _check_maxima_bounds(self):
        """Check that the list of maxima bounds is a list of Tuples of length 2."""
        for index, maxima in enumerate(self.maxima_bounds):
            if len(maxima) != 2:
                raise TypeError(f"Maximum location number {index + 1} is incorrect."
                                "Should be a Tuple[float, float]")

    def __str__(self):
        """String representation of PeakParams, can be copy pasted for instantiation."""
        if self.maxima_bounds[0] == self.peak_bounds:
            return f"PeakParams('{self.name}', {self.peak_bounds})"
        return f"PeakParams('{self.name}', {self.peak_bounds}, {self.maxima_bounds})"


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

    def plot(self, cakes_to_plot: Union[int, List[int]], x_range: Tuple[float, float] = None,
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

    def plot_peak_params(self, peak_params: Union[PeakParams, List[PeakParams]],
                         cakes_to_plot: Union[int, List[int]],
                         x_range: Tuple[float, float] = None, merge_cakes: bool = False,
                         show_points=False):
        if isinstance(cakes_to_plot, int):
            cakes_to_plot = [cakes_to_plot]
        if isinstance(peak_params, PeakParams):
            peak_params = [peak_params]

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
            new_fit.raw_spectrum = self.get_spectrum_subset(cakes, peak.peak_bounds, merge_cakes)
            if merge_cakes:
                new_fit.cake_numbers = [" + ".join(map(str, cakes))]
                new_fit.result = do_pv_fit(new_fit.raw_spectrum, peak.maxima_bounds,
                                           peak.previous_fit_parameters)
            else:
                new_fit.cake_numbers = list(map(str, cakes))
                stacked_spectrum = get_stacked_spectrum(new_fit.raw_spectrum)
                new_fit.result = do_pv_fit(stacked_spectrum, peak.maxima_bounds,
                                           peak.previous_fit_parameters)
            self.fitted_peaks.append(new_fit)
        if self.verbose:
            print("Fitting complete.")

    def get_spectrum_subset(self, cakes: Union[int, List[int]],
                            x_range: Union[None, Tuple[float, float]],
                            merge_cakes: bool) -> np.ndarray:
        """Return spectral intensity as a function of 2-theta for a selected 2-theta range.
        :param cakes: One or more cakes to get the intensity for.
        :param x_range: Limits to the two-theta values returned.
        :param merge_cakes: If more than one cake and True, sum the values of all of the cakes
        else return one column for each cake."""
        if isinstance(cakes, int):
            cakes = [cakes]

        if x_range is None:
            x_range = [0, len(self.spectral_data[:, 0])]

        theta_mask = np.logical_and(self.spectral_data[:, 0] > x_range[0],
                                    self.spectral_data[:, 0] < x_range[1])
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

    def detect_peaks(self, cakes: Union[int, List[int]],
                     x_range: Tuple[float, float] = None) -> List[PeakParams]:
        """
        All parameters in this function should be in units of data points and so
        agnostic to the scale of the dataset being analysed. It will however be affected
        by the density of the data points.
        """
        sub_spectrum = self.get_spectrum_subset(cakes, x_range, merge_cakes=True)

        noise_level = np.percentile(sub_spectrum[:, 1], 20)

        # Detect peaks in signal
        peaks, peak_properties = find_peaks(sub_spectrum[:, 1], height=[None, None],
                                            prominence=[0.6 * noise_level, None], width=[1, None])

        # Separate out singlet and multiplet peaks
        doublet_x_threshold = 15
        doublet_y_threshold = 3
        non_singlet_peaks = []

        for peak_num, peak_index in enumerate(peaks):
            if peak_num + 1 < len(peaks):
                next_peak_index = peaks[peak_num + 1]
                if (next_peak_index - peak_index) < doublet_x_threshold:
                    if np.min(sub_spectrum[peak_index:next_peak_index,
                              1]) > doublet_y_threshold * noise_level:
                        non_singlet_peaks.append(peak_num)
                        non_singlet_peaks.append(peak_num + 1)

        # Build up list of PeakParams
        peak_params = []
        # Convert from data indices to two theta values
        conversion_factor = sub_spectrum[1, 0] - sub_spectrum[0, 0]
        # The offset of the whole spectrum from 0.
        spectrum_offset = sub_spectrum[0, 0]
        # A constant factor determining how wide the peak_bounds of PeakParams are set
        constant_factor = 1.5
        for peak_num, peak_index in enumerate(peaks):
            if peak_num not in non_singlet_peaks:
                half_width = 2 * peak_properties["widths"][peak_num]
                left_offset = np.floor(peak_index - half_width * constant_factor)
                left = left_offset * conversion_factor + spectrum_offset
                right_offset = np.ceil(peak_index + half_width * constant_factor)
                right = right_offset * conversion_factor + spectrum_offset
                peak_params.append(PeakParams(str(peak_num), (round(left, 2), round(right, 2))))

        # Print the PeakParams to std out in a copy/pasteable format.
        print("[", end="")
        for param in peak_params:
            if param != peak_params[-1]:
                print(f"{param},")
            else:
                print(f"{param}]")

        return peak_params


class FittingExperiment:
    """Information about a series of fits to temporally spaced diffraction patterns.
    :ivar spectrum_time: Time between subsequent diffraction patterns.
    :ivar file_string: String used to glob for the diffraction patterns.
    :ivar first_cake_angle:"""
    def __init__(self, spectrum_time: int, file_string: str, first_cake_angle: int,
                 cakes_to_fit: List[int], peak_params: Union[PeakParams, List[PeakParams]],
                 merge_cakes: bool, frames_to_load: List[int] = None):
        self.spectrum_time = spectrum_time
        self.file_string = file_string
        self.first_cake_angle = first_cake_angle
        self.cakes_to_fit = cakes_to_fit
        if isinstance(peak_params, PeakParams):
            peak_params = [peak_params]
        self.peak_params = peak_params
        self.merge_cakes = merge_cakes
        self.frames_to_load = frames_to_load

        self.timesteps: List[FitSpectrum] = []

    def run_analysis(self, reuse_fits=False):
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

            if reuse_fits:
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
