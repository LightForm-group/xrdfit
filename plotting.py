from typing import Tuple, List

import lmfit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rcParams['axes.formatter.useoffset'] = False


def plot_polar_heatmap(num_cakes, rad, z_data, first_cake_angle):
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


def plot_spectrum(data, cakes_to_plot, merge_cakes: bool, show_points: bool,
                  x_range: Tuple[float, float]):
    """Plot a raw spectrum."""
    plt.figure(figsize=(8, 6))
    line_spec = get_line_spec(show_points)
    if merge_cakes:
        plt.plot(data[:, 0], data[:, 1:], line_spec, linewidth=2)
    else:
        for cake_num in cakes_to_plot:
            plt.plot(data[:, 0], data[:, cake_num], line_spec, linewidth=2, label=cake_num)
        plt.legend()

    # Plot formatting
    plt.minorticks_on()
    plt.xlabel(r'Two Theta ($^\circ$)')
    plt.ylabel('Intensity')
    plt.xlim(x_range[0], x_range[1])
    plt.tight_layout()


def plot_peak_params(peak_params):
    for params in peak_params:
        for param in params.maxima_ranges:
            if "min" in param:
                min_x = params.maxima_ranges[param]
                max_x = params.maxima_ranges[param.replace("min", "max")]
                plt.axvspan(min_x, max_x, alpha=0.2, color='grey')
                plt.axvline(min_x, ls="--", color="red")
                plt.axvline(max_x, ls="--", color="green")


def plot_peak_fit(data: np.ndarray, cake_numbers: List[int], fit_result: lmfit.model.ModelResult,
                  fit_name: str):
    """Plot the result of a peak fit as well as the raw data."""
    plt.figure(figsize=(8, 6))

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
    plt.title(fit_name)
    plt.tight_layout()
    plt.show()


def plot_parameter(data: np.ndarray, fit_parameter: str, peak_name: str, show_points: bool):
    """Plot a parameter of a fit against time."""
    line_spec = get_line_spec(show_points)
    plt.plot(data[:, 0], data[:, 1], line_spec)
    plt.xlabel("Time (s)")
    plt.ylabel(fit_parameter.replace("_", " ").title())
    plt.title("Peak {}".format(peak_name))
    plt.show()


def get_line_spec(show_points: bool) -> str:
    """Determine how the data points are shown with and without the raw data."""
    if show_points:
        return "-x"
    return "-"
