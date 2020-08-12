""" This module contain functions for plotting spectral data and the fits to it.
None of these functions should be called directly by users - these functions are called from
plot methods in spectrum_fitting.
"""

import os
import pathlib
from typing import Tuple, List, Union, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# TYPE_CHECKING is False at runtime but allows Type hints in IDE
if TYPE_CHECKING:
    from xrdfit.spectrum_fitting import PeakParams, PeakFit

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rcParams['axes.formatter.useoffset'] = False


def plot_polar_heat_map(num_cakes: int, rad: List[int], z_data: np.ndarray, first_cake_angle: int):
    """Plot a polar heat map using matplotlib.

    :param num_cakes: The number of segments the polar map is divided into.
    :param rad: The radial bin edges.
    :param z_data: A num_cakes by rad shaped array of data to plot.
    :param first_cake_angle: The angle clockwise from vertical at which to label the first cake."""
    degrees_per_cake = 360 / num_cakes
    half_cake_angle = degrees_per_cake / 2

    azm = np.linspace(0, 2 * np.pi, num_cakes + 1)
    rad = np.insert(rad, 0, 0)
    r, theta = np.meshgrid(rad, azm)
    # Offset is anticlockwise regardless of theta direction.
    # Add 90 to theta offset since theta zero defaults to east.
    plt.subplot(projection="polar", theta_direction=-1,
                theta_offset=np.deg2rad(half_cake_angle - first_cake_angle + 90))
    plt.pcolormesh(theta, r, z_data.T)
    plt.plot(azm, r, ls='none')
    plt.grid()
    # Turn on theta grid lines at the cake edges
    plt.thetagrids([theta * 360 / num_cakes for theta in range(num_cakes)], labels=[])
    # Turn off radial grid lines
    plt.rgrids([])
    ax = plt.gca()
    # Put the cake numbers in the right places. Rotation is clockwise in accordance with
    # theta_direction -1 above.
    trans, _, _ = ax.get_xaxis_text1_transform(0)
    for label in range(1, num_cakes + 1):
        ax.text(
            np.deg2rad((label * degrees_per_cake - half_cake_angle)),
            -0.1, label, transform=trans, rotation=0, ha="center", va="center")
    plt.show()


def plot_spectrum(data: np.ndarray, cakes_to_plot: List[int], merge_cakes: bool, show_points: bool,
                  x_range: Union[None, Tuple[float, float]] = None, log_scale=False):
    """Plot a raw spectrum using matplotlib.

    :param data: The data to plot, x_data in column 0, y data in columns 1-N where N is the number
      of cakes in the dataset.
    :param cakes_to_plot: Which cakes (columns of y data) to plot.
    :param merge_cakes: If True plot the sum of the selected cakes as a single line. If False plot
      all selected cakes individually.
    :param show_points: Whether to show data points on the plot.
    :param x_range: If supplied, restricts the x-axis of the plot to this range.
    :param log_scale: If True, plot y axis on log scale. If False use linear scale.
    """
    if show_points:
        line_spec = "-x"
    else:
        line_spec = "-"

    if x_range:
        x_mask = np.logical_and(x_range[0] < data[:, 0], data[:, 0] < x_range[1])
    else:
        x_mask = [True] * data.shape[0]
    if merge_cakes:
        plt.plot(data[x_mask, 0], data[x_mask, 1:], line_spec, linewidth=2)
    else:
        for cake_num in cakes_to_plot:
            plt.plot(data[x_mask, 0], data[x_mask, cake_num], line_spec, linewidth=2,
                     label=cake_num)
        plt.legend()

    # Plot formatting
    plt.minorticks_on()
    plt.xlabel(r'Two Theta ($^\circ$)')
    plt.ylabel('Intensity')
    if x_range:
        plt.xlim(x_range[0], x_range[1])
    if log_scale:
        plt.yscale("log")
    plt.tight_layout()


def plot_peak_params(peak_params: List["PeakParams"], x_range: Tuple[float, float],
                     label_angle: float):
    """A visualisation to show the PeakParams. Peak bounds are indicated by a shaded grey area.
    Maxima bounds are shown by a dashed green line for the min bound and a dashed red line for
    the max bound. This method is called with an active plot environment and plots the peak
    params on top.

    :param peak_params: The peak params to plot.
    :param x_range: If supplied, restricts the x-axis of the plot to this range.
    :param label_angle: If supplied, the angle to rotate the maxima labels.
    """
    for params in peak_params:
        bounds_min = params.peak_bounds[0]
        bounds_max = params.peak_bounds[1]
        range_center = (bounds_min + bounds_max) / 2
        plt.axvline(bounds_min, ls="-", lw=1, color="grey")
        plt.axvline(bounds_max, ls="-", lw=1, color="grey")
        plt.axvspan(bounds_min, bounds_max, alpha=0.2, color='grey', hatch="/")
        for maximum in params.maxima:
            min_x = maximum.bounds[0]
            max_x = maximum.bounds[1]
            center = (min_x + max_x) / 2
            plt.axvline(min_x, ls="--", color="green")
            plt.axvline(max_x, ls="--", color="red")
            if x_range[0] < range_center < x_range[1]:
                plt.text(center, plt.ylim()[1], maximum.name, ha="center", va="bottom",
                         fontsize=matplotlib.rcParams["axes.titlesize"] * 0.8, rotation=label_angle)
        plt.xlim(x_range)


def plot_peak_fit(peak_fit: "PeakFit", time_step: str = None, file_name: str = None,
                  title: str = None, label_angle: float = None, log_scale=False):
    """Plot the result of a peak fit as well as the raw data.

    :param peak_fit: The result of a peak fit
    :param time_step: If provided, used to generate the title of the plot.
    :param file_name: If provided used as a on disk location to save the plot.
    :param title: If provided, can be used to override the auto generated plot title.
    :param label_angle: The angle to rotate maxima labels.
    :param log_scale: Whether to plot the y axis on a log or linear scale.
    """
    data = peak_fit.raw_spectrum
    # First plot the raw data
    for index, cake_num in enumerate(peak_fit.cake_numbers):
        plt.plot(data[:, 0], data[:, index + 1], 'x', ms=10, mew=3, label=f"Cake {cake_num}")

    title_size = matplotlib.rcParams["axes.titlesize"]

    # Now plot the fit
    x_data = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
    y_fit = peak_fit.result.model.eval(peak_fit.result.params, x=x_data)
    plt.plot(x_data, y_fit, 'k--', lw=1, label="Fit")
    # Do all the ancillaries to make the plot look good.
    plt.minorticks_on()
    plt.tight_layout()
    plt.xlabel(r'Two Theta ($^\circ$)')
    plt.ylabel('Intensity')
    plt.legend()
    if title is not None:
        plt.title(title, va="bottom", fontsize=title_size, pad=title_size * 1.8)
    elif time_step is not None:
        plt.title(f'Fit at t = {time_step}', va="bottom", fontsize=title_size, pad=title_size * 1.8)

    for index, maxima_name in enumerate(peak_fit.maxima_names):
        maxima_center = peak_fit.result.params[f"maximum_{index}_center"]
        plt.text(maxima_center, plt.ylim()[1] * 1.05, maxima_name, horizontalalignment="center",
                 fontsize=title_size * 0.8, rotation=label_angle)
    if log_scale:
        plt.yscale("log")
    plt.tight_layout()
    if file_name:
        file_name = pathlib.Path(file_name)
        if not file_name.parent.exists():
            os.makedirs(file_name.parent)
        plt.savefig(file_name)
    else:
        plt.show()


def plot_parameter(data: np.ndarray, fit_parameter: str, show_points: bool,
                   show_error: bool, scale_by_error: bool = False, log_scale=False):
    """Plot a parameter of a fit against time.

    :param data: The data to plot, x data in the first column, y data in the second column and
      the y error in the third column.
    :param fit_parameter: The name of the parameter being plotted, used to generate the y-axis label
    :param show_points: Whether to show data points on the plot.
    :param show_error: Whether to show error bars on the plot.
    :param scale_by_error: If True auto scale the y-axis to the range of the error bars. If False,
      auto scale the y-axis to the range of the data.
    :param log_scale: Whether to plot the y-axis on a log or linear scale.
    """
    no_covar_mask = data[:, 2] == 0
    covar_mask = [not value for value in no_covar_mask]
    # Plotting the data
    plt.plot(data[:, 0], data[:, 1], "-", mec="red")
    # Save the y-range to reapply later if wanted
    data_y_range = plt.ylim()
    if show_points:
        plt.plot(data[covar_mask, 0], data[covar_mask, 1], "x", mec="blue")
        plt.plot(data[no_covar_mask, 0], data[no_covar_mask, 1], "^", mec="blue")
    # Plotting the error bars
    if show_error:
        plt.fill_between(data[:, 0], data[:, 1] - data[:, 2], data[:, 1] + data[:, 2], alpha=0.3)
        plt.plot(data[:, 0], data[:, 1] - data[:, 2], "--", lw=0.5, color='gray')
        plt.plot(data[:, 0], data[:, 1] + data[:, 2], "--", lw=0.5, color='gray')

    if not scale_by_error:
        plt.ylim(data_y_range)
    if log_scale:
        plt.yscale("log")

    plt.xlabel("Time (s)")
    plt.ylabel(fit_parameter.replace('_', ' ').title())
    plt.show()
