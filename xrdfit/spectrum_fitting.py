"""This module contains the main fitting functions of xrdfit."""

import bz2
import glob
import time
from typing import List, Tuple, Union

import numpy as np
import lmfit
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import dill

import xrdfit.plotting as plotting
from xrdfit.pv_fit import do_pv_fit


class PeakParams:
    """An object containing information about a peak and its maxima.

    :ivar peak_bounds: Where in the spectrum the peak begins and ends. The fit will be done over
      this region.
    :ivar maxima_names: A name for each of the maxima.
    :ivar maxima_bounds: If there is more than one maxima, a bounding box for each peak center.
    :ivar previous_fit_parameters: If running multiple fits over time using a
      :class:`FitExperiment`, the result of the previous fit.
    :ivar peak_name: The name of the peak, made from compounding the maxima names.
    """
    def __init__(self, peak_bounds: Tuple[float, float], maxima_names: Union[str, List[str]],
                 maxima_bounds: List[Tuple[float, float]] = None):
        """
        :param peak_bounds: Where in the spectrum the peak begins and ends. The fit will be
          done over this region.
        :param maxima_names: A name for each of the maxima.
        :param maxima_bounds: If there is more than one maxima, a bounding box for each peak center.
        """
        self.peak_bounds = peak_bounds
        if isinstance(maxima_names, str):
            maxima_names = [maxima_names]
        self.maxima_names = maxima_names
        if maxima_bounds:
            self.maxima_bounds = maxima_bounds
            self._check_maxima_bounds()
        else:
            self.maxima_bounds = [peak_bounds]
        self._check_maxima_names()
        self.peak_name = " ".join(maxima_names)

        self.previous_fit_parameters: Union[lmfit.Parameters, None] = None

    def __str__(self) -> str:
        """String representation of PeakParams. Can be copy pasted for instantiation of new
        PeakParams."""
        if self.maxima_bounds[0] == self.peak_bounds:
            return f"PeakParams({self.peak_bounds}, '{self.maxima_names}')"
        return f"PeakParams({self.peak_bounds}, '{self.maxima_names}', {self.maxima_bounds})"

    def _check_maxima_bounds(self):
        """Check that the list of maxima bounds is a list of Tuples of length 2."""
        for index, maxima in enumerate(self.maxima_bounds):
            if len(maxima) != 2:
                raise TypeError(f"Maximum location number {index + 1} is incorrect."
                                "Should be a Tuple[float, float]")

    def _check_maxima_names(self):
        """Check that there is one name for each set of maxima bounds."""
        if len(self.maxima_names) != len(self.maxima_bounds):
            raise TypeError(f"Number of maxima does not match number of maxima names."
                            f"{len(self.maxima_names)} names are specified and "
                            f"{len(self.maxima_bounds)} maxima are specified.")

    def set_previous_fit(self, fit_params: lmfit.Parameters):
        """When passing the result of a previous fit to the next fit, the peak center may drift
        over time. The limits for the next fit must be reset otherwise it will always have the
        limits from the first fit.

        :param fit_params: The final parameters of the previous fit.
        """
        for parameter in fit_params.values():
            if "center" in parameter.name:
                center_value = parameter.value
                center_range = parameter.max - parameter.min
                center_min = center_value - (center_range / 2)
                center_max = center_value + (center_range / 2)
                fit_params.add(parameter.name, value=center_value, min=center_min,
                               max=center_max)
        self.previous_fit_parameters = fit_params

    def adjust_peak_bounds(self, fit_params: lmfit.Parameters):
        """Adjust peak bounds to re-center the peak in the peak bounds.

        :param fit_params: The final parameters of the previous fit.
        """
        centers = []
        for name in fit_params:
            if "center" in name:
                centers.append(fit_params[name].value)
        center = sum(centers) / len(centers)

        bound_width = self.peak_bounds[1] - self.peak_bounds[0]
        self.peak_bounds = (center - (bound_width / 2), center + (bound_width / 2))

    def adjust_maxima_bounds(self, fit_params: lmfit.Parameters):
        """Adjust maxima bounds to re-center the maximum in the maximum bounds.

        :param fit_params: The final parameters of the previous fit.
        """
        peak_centers = []
        for param in fit_params:
            if "center" in param:
                peak_centers.append(fit_params[param].value)

        new_maxima_bounds = []
        for center, maximum_bounds in zip(peak_centers, self.maxima_bounds):
            maximum_bound_width = maximum_bounds[1] - maximum_bounds[0]
            lower_bound = center - maximum_bound_width / 2
            upper_bound = center + maximum_bound_width / 2
            new_maxima_bounds.append((lower_bound, upper_bound))
        self.maxima_bounds = new_maxima_bounds


class PeakFit:
    """An object containing data on the fit to a peak.

    :ivar name: The name of the peak.
    :ivar maxima_names: The names of the maxima in the peak.
    :ivar raw_spectrum: The raw data to which the fit is made.
    :ivar result: The lmfit result of the fit.
    :ivar cake_numbers: The cake number each column in raw_spectrum refers to.
    """
    def __init__(self, peak_params: PeakParams):
        """
        :param peak_params: A PeakParams object describing the peak to be fitted.
        """
        self.name = peak_params.peak_name
        self.maxima_names = peak_params.maxima_names
        self.raw_spectrum: Union[None, np.ndarray] = None
        self.result: Union[None, lmfit.model.ModelResult] = None
        self.cake_numbers: List[int] = []

    def plot(self, time_step: str = None, file_name: str = None, title: str = None):
        """ Plot the raw spectral data and the fit.
        :param time_step: If provided, a time step used to generate the title of the plot.
        :param file_name: If provided, save the plot to this file as well as displaying it.
        :param title: If provided, override the autogenerated plot title with this title.
        """
        if self.raw_spectrum is None:
            print("Cannot plot fit peak as fitting has not been done yet.")
        else:
            plotting.plot_peak_fit(self.raw_spectrum, self.cake_numbers, self.result, self.name,
                                   time_step, file_name, title)


class FitSpectrum:
    """An object that stores data about a spectrum and its fitted peaks.

    :ivar verbose: Whether or not to print fit status to the console.
    :ivar first_cake_angle: The angle of the first cake in the data file in degrees
      clockwise from North.
    :ivar fitted_peaks: Fits to peaks in the spectrum.
    :ivar num_evaluations: A dict of peak names and how many iterations the fit took to converge.
    :ivar fit_time: A dict of peak names and the time taken to evaluate that fit.
    :ivar spectral_data: Data for the whole diffraction pattern.
    """
    def __init__(self, file_path: str, first_cake_angle: int = 90, verbose: bool = True):
        """
        :param file_path: The path of the file containing scattering data to load.
        :param first_cake_angle: The angle of the first cake in the data file in degrees
          clockwise from North.
        :param verbose: Whether or not to print fit status to the console
        """
        self.verbose = verbose
        self.first_cake_angle = first_cake_angle
        self.fitted_peaks: List[PeakFit] = []
        self.num_evaluations = {}
        self.fit_time = {}

        self.spectral_data = pd.read_table(file_path).to_numpy()
        if self.verbose:
            print("Diffraction pattern successfully loaded from file.")

    def plot_polar(self):
        """Plot the whole diffraction pattern on polar axes."""
        with np.errstate(divide='ignore'):
            z_data = np.log10(self.spectral_data[:, 1:])
        rad = self.spectral_data[:, 0]
        num_cakes = z_data.shape[1]
        plotting.plot_polar_heat_map(num_cakes, rad, z_data, self.first_cake_angle)

    def highlight_cakes(self, cakes: Union[int, List[int]]):
        """Plot a circular map of diffraction pattern with the selected cakes highlighted.

        :param cakes: The cake numbers to be highlighted.
        """
        num_cakes = self.spectral_data.shape[1] - 1
        z_data = np.zeros((1, self.spectral_data.shape[1] - 1))
        for cake_num in cakes:
            z_data[0, cake_num - 1] = 1
        rad = [0, 1]
        plotting.plot_polar_heat_map(num_cakes, rad, z_data, self.first_cake_angle)

    def plot(self, cakes_to_plot: Union[int, List[int]], x_range: Tuple[float, float] = None,
             merge_cakes: bool = False, show_points=False):
        """Plot the intensity as a function of two theta for a given cake.

        :param cakes_to_plot: The numbers of one or more cakes to plot.
        :param x_range: If supplied, restricts the x-axis of the plot to this range.
        :param merge_cakes: If True plot the sum of the selected cakes as a single line. If False
          plot all selected cakes individually.
        :param show_points: Whether to show data points on the plot.
        """
        if isinstance(cakes_to_plot, int):
            cakes_to_plot = [cakes_to_plot]
        # Get the data to plot
        if merge_cakes:
            data = self._get_spectrum_subset(cakes_to_plot, x_range, True)
        else:
            data = self.spectral_data
        plotting.plot_spectrum(data, cakes_to_plot, merge_cakes, show_points, x_range)
        plt.show()

    def plot_fit(self, fit_name: str, time_step: str = None, file_name: str = None):
        """Plot the result of a fit and the raw data.

        :param fit_name: The name of the fit to plot.
        :param time_step: If provided, the time_step of the fit which will be added to the title.
        :param file_name: If provided, the stub of the file name to write the plot to, if not
          provided, the plot will be displayed on screen.
        """
        fit = self.get_fit(fit_name)
        fit.plot(time_step, file_name)

    def plot_peak_params(self, peak_params: Union[PeakParams, List[PeakParams]],
                         cakes_to_plot: Union[int, List[int]],
                         x_range: Tuple[float, float] = None, merge_cakes: bool = False,
                         show_points=False):
        """Plot a visualisation of the provided :class:`PeakParams` over the raw data.

        :param peak_params: The :class:`PeakParams` to plot.
        :param cakes_to_plot: The numbers of one or more cakes to plot to raw data for.
        :param x_range: If supplied, restricts the x-axis of the plot to this range.
        :param merge_cakes: If True plot the sum of the selected cakes as a single line. If False
          plot all selected cakes individually.
        :param show_points: Whether to show data points on the plot.
        """
        if isinstance(cakes_to_plot, int):
            cakes_to_plot = [cakes_to_plot]
        if isinstance(peak_params, PeakParams):
            peak_params = [peak_params]

        # Get the data to plot
        if merge_cakes:
            data = self._get_spectrum_subset(cakes_to_plot, x_range, True)
        else:
            data = self.spectral_data
        if x_range is None:
            bounds = [param.peak_bounds for param in peak_params]
            min_bound = min(bounds)[0]
            max_bound = max(bounds, key=lambda x: x[1])[1]
            bound_range = max_bound - min_bound
            padding = bound_range / 10
            x_range = (min_bound - padding, max_bound + padding)
        plotting.plot_spectrum(data, cakes_to_plot, merge_cakes, show_points, x_range)
        plotting.plot_peak_params(peak_params, x_range)
        plt.show()

    def fit_peaks(self, peak_params: Union[PeakParams, List[PeakParams]],
                  cakes: Union[int, List[int]], merge_cakes: bool = False, debug: bool = False):
        """Attempt to fit peaks within the ranges specified by :class:`PeakParams`.

        :param peak_params: A list of :class:`PeakParams` describing the peaks to be fitted.
        :param cakes: Which cakes to fit.
        :param merge_cakes: If True and multiple cakes are specified then sum the cakes before
          fitting. Else do the fit to multiple cakes simultaneously.
        :param debug: Whether to show debug info for slow fits.
        """
        self.fitted_peaks = []
        if isinstance(cakes, int):
            cakes = [cakes]
        if isinstance(peak_params, PeakParams):
            peak_params = [peak_params]

        self.fit_time = {peak.peak_name: 0 for peak in peak_params}
        self.num_evaluations = {peak.peak_name: 0 for peak in peak_params}

        for peak in peak_params:
            new_fit = PeakFit(peak)
            new_fit.raw_spectrum = self._get_spectrum_subset(cakes, peak.peak_bounds, merge_cakes)
            start = time.perf_counter()
            if merge_cakes:
                new_fit.cake_numbers = [" + ".join(map(str, cakes))]
                new_fit.result = do_pv_fit(new_fit.raw_spectrum, peak.maxima_bounds,
                                           peak.previous_fit_parameters)
            else:
                new_fit.cake_numbers = list(map(str, cakes))
                stacked_spectrum = _get_stacked_spectrum(new_fit.raw_spectrum)
                new_fit.result = do_pv_fit(stacked_spectrum, peak.maxima_bounds,
                                           peak.previous_fit_parameters)
            fit_time = time.perf_counter() - start
            self.fitted_peaks.append(new_fit)
            # Debug for slow fits
            if new_fit.result.nfev > 500 and debug:
                print(peak.peak_name)
                print(new_fit.result.init_params)
                print(new_fit.result.params)
                new_fit.result.plot_fit(show_init=True, numpoints=500)
                plt.show()

            # Accounting
            self.num_evaluations[peak.peak_name] = new_fit.result.nfev
            self.fit_time[peak.peak_name] = fit_time
            
        if self.verbose:
            print("Fitting complete.")

    def _get_spectrum_subset(self, cakes: Union[int, List[int]],
                             x_range: Union[None, Tuple[float, float]],
                             merge_cakes: bool) -> np.ndarray:
        """Return spectral intensity as a function of 2-theta for a selected 2-theta range.

        :param cakes: One or more cakes to get the intensity for.
        :param x_range: Limits to the two-theta values returned.
        :param merge_cakes: If more than one cake and True, sum the values of all of the cakes
          else return one column for each cake.
        """
        if isinstance(cakes, int):
            cakes = [cakes]

        if x_range is None:
            x_range = [-np.inf, np.inf]

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
        """Get a :class:`PeakFit` by name.

        :param name: The name of the :class:`PeakFit` to get.
        """
        for fit in self.fitted_peaks:
            if fit.name == name:
                return fit
            for maximum_name in fit.maxima_names:
                if maximum_name == name:
                    return fit
        raise KeyError(f"Fit: '{name}' not found")


class FitReport:
    """Some details about the performance of a :class:`FitExperiment`.

    :ivar fit_time: The cumulative time taken to fit each peak in the spectrum.
    :ivar num_evaluations: The number of evaluations taken for the fit to minimise for
      each peak in the spectrum.
    :ivar num_time_steps: The number of time steps in the :class:`FitExperiment`.
    """
    def __init__(self, peak_names: List[str]):
        """
        :param peak_names: The names of the peaks being fitted in the :class:`FitExperiment`.
        """
        self.fit_time: dict = {peak_name: 0 for peak_name in peak_names}
        self.num_evaluations: dict = {peak_name: [] for peak_name in peak_names}
        self.num_time_steps = None

    def print(self, evaluation_threshold: int = 500, detailed=False):
        """Print the fit report to the console.

        :param evaluation_threshold: The number of fitting iterations that the triggers the report
          to warn about slow fitting.
        :param detailed: If True, also report the time taken to do the fits.
        """
        slow_fits = {}
        # Work out if any of the fits are slower than the evaluation threshold
        for name, num_evaluations in self.num_evaluations.items():
            num_evaluations = np.array(num_evaluations)
            slow_fits[name] = np.sum(num_evaluations > evaluation_threshold)
        if sum(slow_fits.values()):
            print(f"The following fits took over {evaluation_threshold} fitting iterations. "
                  f"The quality of these fits should be checked.")
            for peak_name, num_evaluations in slow_fits.items():
                if num_evaluations > 0:
                    percentage = num_evaluations / self.num_time_steps * 100
                    print(f"{percentage:2.1f}% of fits for peak {peak_name}")

            if detailed:
                print(f"\nFit times:")
                for peak_name, fit_time in self.fit_time.items():
                    print(f"{fit_time:2.1f} s: {peak_name}")


class FitExperiment:
    """Information about a series of fits to temporally spaced diffraction patterns.

    :ivar spectrum_time: Time between subsequent diffraction patterns.
    :ivar file_string: String used to glob for the diffraction patterns.
    :ivar first_cake_angle: The angle of the first cake in the data file in degrees
      clockwise from North.
    :ivar cakes_to_fit: Which of the cakes in the diffraction pattern to fit.
    :ivar peak_params: The provided :class:`PeakParams` to use for fitting.
    :ivar merge_cakes: If multiple cakes are requested, whether to merge them or fit them \
      separately.
    :ivar frames_to_load: If specified, which time steps to fit.
    :ivar time_steps: A list of :class:`FitSpectrum` one for each time step.
    :ivar fit_report: A :class:`FitReport` for this :class`FittingExperiment`.
    """
    def __init__(self, spectrum_time: int, file_string: str, first_cake_angle: int,
                 cakes_to_fit: List[int], peak_params: Union[PeakParams, List[PeakParams]],
                 merge_cakes: bool, frames_to_load: List[int] = None):
        """
        :param spectrum_time: Time between subsequent diffraction patterns.
        :param file_string: String used to glob for the diffraction patterns.
        :param first_cake_angle: The angle of the first cake in the data file in degrees \
          clockwise from North.
        :param cakes_to_fit: Which of the cakes in the diffraction pattern to fit.
        :param peak_params: The provided :class:`PeakParams` to use for fitting.
        :param merge_cakes: If multiple cakes are requested, whether to merge them or fit them \
          separately.
        :param frames_to_load: If specified, which time steps to fit.
        """
        self.spectrum_time = spectrum_time
        self.file_string = file_string
        self.first_cake_angle = first_cake_angle
        self.cakes_to_fit = cakes_to_fit
        if isinstance(peak_params, PeakParams):
            peak_params = [peak_params]
        self.peak_params = peak_params
        self.merge_cakes = merge_cakes
        self.frames_to_load = frames_to_load

        self.time_steps: List[FitSpectrum] = []
        self.fit_report = FitReport([peak.peak_name for peak in peak_params])

    def run_analysis(self, reuse_fits=False, debug: bool = False):
        """Run a fit over multiple diffraction patterns.

        :param reuse_fits: If True, use the result of one time step to provide the initial
          parameters for the next fit. If False, guess the initial fit parameters from the data
          at each time step.
        :param debug: If True, print a short report each time a fit takes more than 500 time steps.
          The report gives the initial parameters, the final parameters and a plot of these with
          the raw data.
        """
        if self.frames_to_load:
            file_list = [self.file_string.format(number) for number in self.frames_to_load]
        else:
            file_list = sorted(glob.glob(self.file_string))
            if len(file_list) == 0:
                raise FileNotFoundError(f"No files found with file stub: '{self.file_string}'")

        self.fit_report.num_time_steps = len(file_list)

        print(f"Processing {len(file_list)} diffraction patterns.")
        for file_path in tqdm(file_list):
            spectral_data = FitSpectrum(file_path, self.first_cake_angle, verbose=False)
            spectral_data.fit_peaks(self.peak_params, self.cakes_to_fit, self.merge_cakes, debug)
            self.time_steps.append(spectral_data)

            # Prepare the PeakParams for the next time step.
            for peak_fit, peak_params in zip(spectral_data.fitted_peaks, self.peak_params):
                if reuse_fits:
                    # Check signal to noise is good enough to reuse the params
                    if _check_snr(peak_fit.result.params):
                        peak_params.set_previous_fit(peak_fit.result.params)
                # Move maxima bounds and peak bounds to keep shifting peaks centered in the bounds.
                peak_params.adjust_maxima_bounds(peak_fit.result.params)
                peak_params.adjust_peak_bounds(peak_fit.result.params)
            self._update_fit_report(spectral_data)

        print("Analysis complete.")
        self.fit_report.print()

    def peak_names(self) -> List[str]:
        """List the names of the peaks specified for fitting in the PeakParams."""
        return [peak.peak_name for peak in self.peak_params]

    def fit_parameters(self, peak_name: str) -> List[str]:
        """List the names of the parameters of the fit for a specified peak.

        :param peak_name: The peak to list the parameters of.
        """
        if self.time_steps:
            return list(self.time_steps[0].get_fit(peak_name).result.values.keys())

    def get_fit_parameter(self, peak_name: str, fit_parameter: str) -> Union[None, np.ndarray]:
        """Get the raw values and error of a fitting parameter over time.

        :param peak_name: The name of the peak to get the data for.
        :param fit_parameter: The name of the fit parameter to get the data for.
        :returns: A NumPy array with x data in the first column, y data in the second column and
          the y-error in the third column.
        """
        if peak_name not in [peak.peak_name for peak in self.peak_params]:
            print(f"Peak '{peak_name}' not found in fitted peaks.")
            return None
        if fit_parameter not in self.fit_parameters(peak_name):
            print(f"Unknown fit parameter {fit_parameter} for peak {peak_name}")
            return None

        parameters = []
        for time_step in self.time_steps:
            peak_fit = time_step.get_fit(peak_name)
            parameters.append(peak_fit.result.params[fit_parameter])
        if self.frames_to_load:
            x_data = np.array(self.frames_to_load) * self.spectrum_time
        else:
            x_data = (np.arange(len(parameters)) + 1) * self.spectrum_time
        # It is possible that leastsq can't invert the curvature matrix so cant provide error
        # estimates. In these cases stderr is given as None.
        errors = [parameter.stderr if parameter.stderr else 0 for parameter in parameters]
        values = [parameter.value for parameter in parameters]
        data = np.vstack((x_data, values, errors)).T
        return data

    def plot_fit_parameter(self, peak_name: str, fit_parameter: str, show_points=False,
                           show_error=True, scale_by_error: bool = False):
        """Plot a named parameter of a fit as a function of time.

        :param peak_name: The name of the fit to plot.
        :param fit_parameter: The name of the fit parameter to plot.
        :param show_points: Whether to show data points on the plot.
        :param show_error: Whether to show the y-error as a shaded area on the plot.
        :param scale_by_error: If False the y-axis will be scaled to fit the data values. If True
          the y-axis will be scaled to fit the error values.
        """
        data = self.get_fit_parameter(peak_name, fit_parameter)
        if data is not None:
            plotting.plot_parameter(data, fit_parameter, peak_name, show_points, show_error,
                                    scale_by_error)

    def plot_fits(self, num_time_steps: int = 5, peak_names: Union[List[str], str] = None,
                  time_steps: List[int] = None, file_name: str = None):
        """Plot the calculated fits to the data for this :class:`FitExperiment` instance.

        :param num_time_steps: The number of time_steps to plot fits for. The function will plot
          this many time_steps, evenly spaced over the whole dataset. This value is ignored if
          `time_steps` is specified.
        :param peak_names: The name of the peak to fit. If not specified, will plot all fitted
          peaks.
        :param time_steps: If provided, a list of time_steps to plot the fits for.
        :param file_name: If provided, outputs the plot to an image file with file_name as the
          image name stub.
        """
        if time_steps is None:
            time_steps = self._calculate_time_steps(num_time_steps)
        if peak_names is None:
            peak_names = self.peak_names()
        if isinstance(peak_names, str):
            peak_names = [peak_names]
        for time_step in time_steps:
            for name in peak_names:
                if file_name:
                    output_name = f"../plots/{file_name}_{name}_{time_step :04d}.png"
                else:
                    output_name = None
                self.time_steps[time_step].plot_fit(name, str(time_step), output_name)

    def _calculate_time_steps(self, num_time_steps: int) -> List[int]:
        """Work out which time_steps to plot.

        :param num_time_steps: The total number of timesteps to plot.
        """
        time_steps = np.linspace(0, len(self.time_steps) - 1, num_time_steps)
        # Remove duplicate values
        time_steps = list(dict.fromkeys(np.round(time_steps)).keys())
        time_steps = [int(i) for i in time_steps]

        return time_steps

    def save(self, file_name: str):
        """Dump the data and all fits to a compressed binary file using dill.

        :param file_name: The name of the file to save the data to.
        """
        print("Saving data to dump file.")
        with bz2.open(file_name, 'wb') as output_file:
            dill.dump(self, output_file)
        print("Data successfully saved to dump file.")

    def _update_fit_report(self, spectral_data: FitSpectrum):
        """Update the fit report with stats from the fitting.

        :param spectral_data: The Spectrum which was just fitted.
        """
        for peak_name, fit_time in spectral_data.fit_time.items():
            self.fit_report.fit_time[peak_name] += fit_time
        for peak_name, num_evaluations in spectral_data.num_evaluations.items():
            self.fit_report.num_evaluations[peak_name].append(num_evaluations)


def _get_stacked_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Take an number of observations from N different cakes and stack them vertically into a 2
    column wide array.

    :param spectrum: The spectral data to manipulate.
    """
    stacked_data = spectrum[:, 0:2]
    spectrum_columns = spectrum.shape[1]
    for column_num in range(2, spectrum_columns):
        stacked_data = np.vstack(
            (stacked_data, spectrum[np.ix_([True] * spectrum.shape[0], [0, column_num])]))
    stacked_data = stacked_data[stacked_data[:, 0].argsort()]
    return stacked_data


def load_dump(file_name: str) -> FitExperiment:
    """ Load a FittingExperiment object saved using the FittingExperiment.save() method.

    :param file_name: The path of the file to load the data from.
    """
    print("Loading data from dump file.")
    with bz2.open(file_name, "rb") as input_file:
        data = dill.load(input_file)
        print("Data successfully loaded from dump file.")
        return data


def _check_snr(params: lmfit.Parameters) -> bool:
    """ Iterate through params and check whether any of the maxima in params have a low
    signal to noise ratio.

    :param params: The parameters to check.
    :return True if any fit has an signal to noise ratio of less than 2 else False.
    """
    good_snr = True
    for param_name in params:
        if "snr" in param_name:
            if params[param_name].value < 2:
                good_snr = False
    return good_snr
