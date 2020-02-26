""" This module contain functions for plotting spectral data and the fits to it.
None of these functions should be called directly by users - these functions are called from
plot methods in spectrum_fitting.
"""

import os
import pathlib
from typing import Tuple, List, Union, TYPE_CHECKING

import lmfit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# TYPE_CHECKING is False at runtime but allows Type hints in IDE
if TYPE_CHECKING:
    from xrdfit.spectrum_fitting import PeakParams

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rcParams['axes.formatter.useoffset'] = False


def plot_polar_heatmap(num_cakes: int, rad: List[int], z_data: np.ndarray, first_cake_angle: int):
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
    degrees_per_cake = 360/num_cakes
    half_cake = degrees_per_cake / 2
    for label in range(1, num_cakes + 1):
        ax.text(np.deg2rad(label * degrees_per_cake - 90 - half_cake + first_cake_angle), -0.1,
                label, transform=trans, rotation=0, ha="center", va="center")
    plt.show()


def plot_spectrum(data: np.ndarray, cakes_to_plot: List[int], merge_cakes: bool, show_points: bool,
                  x_range: Union[None, Tuple[float, float]] = None):
    """Plot a raw spectrum."""
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
    plt.tight_layout()


def plot_peak_params(peak_params: List["PeakParams"], x_range: Tuple[float, float]):
    """Plot a dashed line at the maximum and minimum extent of the provided PeakParams and shade
    the contained area."""
    for params in peak_params:
        bounds_min = params.peak_bounds[0]
        bounds_max = params.peak_bounds[1]
        range_center = (bounds_min + bounds_max) / 2
        plt.axvline(bounds_min, ls="-", lw=1, color="grey")
        plt.axvline(bounds_max, ls="-", lw=1, color="grey")
        plt.axvspan(bounds_min, bounds_max, alpha=0.2, color='grey', hatch="/")
        for param in params.maxima_bounds:
            min_x = param[0]
            max_x = param[1]
            plt.axvline(min_x, ls="--", color="green")
            plt.axvline(max_x, ls="--", color="red")
        bottom, top = plt.ylim()
        if x_range[0] < range_center < x_range[1]:
            plt.text(range_center, top, params.name, size=20, ha="center", va="bottom")
        plt.xlim(x_range)


def plot_peak_fit(data: np.ndarray, cake_numbers: List[int], fit_result: lmfit.model.ModelResult,
                  fit_name: str, timestep: str = None, file_name: str = None):
    """Plot the result of a peak fit as well as the raw data."""

    # First plot the raw data
    for index, cake_num in enumerate(cake_numbers):
        plt.plot(data[:, 0], data[:, index + 1], 'x', ms=10, mew=3, label=f"Cake {cake_num}")

    # Now plot the fit
    x_data = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
    y_fit = fit_result.model.eval(fit_result.params, x=x_data)
    plt.plot(x_data, y_fit, 'k--', lw=1, label="Fit")
    plt.minorticks_on()
    plt.tight_layout()
    plt.xlabel(r'Two Theta ($^\circ$)')
    plt.ylabel('Intensity')
    plt.legend()
    if timestep:
        fit_name = f'Peak "{fit_name}" at t = {timestep}'
    plt.title(fit_name)
    plt.tight_layout()
    if file_name:
        file_name = pathlib.Path(file_name)
        if not file_name.parent.exists():
            os.makedirs(file_name.parent)
        plt.savefig(file_name)
    else:
        plt.show()
    plt.close()


def plot_parameter(data: np.ndarray, fit_parameter: str, peak_name: str, show_points: bool,
                   show_error: bool, y_range: Union[None, Tuple[float, float]]):
    """Plot a parameter of a fit against time.
    The data array contains x data in the first column, y data in the second column and the y
    error in the third column.
    """
    no_covar_mask = data[:, 2] == 0
    covar_mask = [not value for value in no_covar_mask]
    # Plotting the data
    plt.plot(data[:, 0], data[:, 1], "-", mec="red")
    # Save the y-range to reapply later because the error bars can make it go crazy
    current_y_range = plt.ylim()
    if show_points:
        plt.plot(data[covar_mask, 0], data[covar_mask, 1], "x", mec="blue")
        plt.plot(data[no_covar_mask, 0], data[no_covar_mask, 1], "^", mec="blue")
    # Plotting the error bars
    if show_error:
        plt.fill_between(data[:, 0], data[:, 1] - data[:, 2], data[:, 1] + data[:, 2], alpha=0.3)
        plt.plot(data[:, 0], data[:, 1] - data[:, 2], "--", lw=0.5, color='gray')
        plt.plot(data[:, 0], data[:, 1] + data[:, 2], "--", lw=0.5, color='gray')

    if y_range:
        plt.ylim(y_range)
    else:
        plt.ylim(current_y_range)

    plt.xlabel("Time (s)")
    plt.ylabel(fit_parameter.replace("_", " ").title())
    plt.title("Peak {}".format(peak_name))
    plt.show()
