""" This module contain functions implementing the Pseudo-Voigt fit in lmfit.
None of these functions should be called directly by users - these functions are called from
methods in spectrum_fitting.
"""

from typing import List, Tuple, TYPE_CHECKING

import lmfit
import numpy as np

if TYPE_CHECKING:
    from spectrum_fitting import PeakParams, MaximumParams


def do_pv_fit(peak_data: np.ndarray, peak_param: "PeakParams") -> lmfit.model.ModelResult:
    """Pseudo-Voigt fit to the lattice plane peak intensity.

    :param peak_data: The data to be fitted, two theta values (x-data) in column 0 and intensity
      (y-data) in column 1.
    :param peak_param: A PeakParams object describing the peak to be fitted.
    """
    model = None
    num_maxima = len(peak_param.maxima)

    # Add one peak to the model for each maximum
    for maxima_num in range(num_maxima):
        prefix = f"maximum_{maxima_num}_"
        if model:
            model += lmfit.models.PseudoVoigtModel(prefix=prefix)
        else:
            model = lmfit.models.PseudoVoigtModel(prefix=prefix)
    model += lmfit.Model(lambda x, background: background)

    two_theta = peak_data[:, 0]
    intensity = peak_data[:, 1]

    new_fit_parameters = guess_params(model, peak_param.previous_fit_parameters, two_theta,
                                      intensity, peak_param.maxima)
    # We can't use special characters in param names so have to save the user provided
    # name in user_data.
    for parameter in new_fit_parameters:
        if parameter != "background":
            parameter_num = int(parameter.split("_")[1])
            new_fit_parameters[parameter].user_data = peak_param.maxima[parameter_num].name

    fit_result = model.fit(intensity, new_fit_parameters, x=two_theta)

    return fit_result


def guess_params(model: lmfit.Model, old_fit_params: lmfit.Parameters,
                 x_data: np.ndarray, y_data: np.ndarray,
                 maxima_params: List["MaximumParams"]) -> lmfit.Parameters:
    """Given a dataset and some information about where the maxima are, guess some good initial
    values for the Pseudo-Voigt fit.

    :param model:  The lmfit Model to guess the params for.
    :param old_fit_params: Any params that are to be passed on from a previous fit
    :param x_data: The x data to be fitted.
    :param y_data: The y data to be fitted.
    :param maxima_params: The MaximaParams specified by the user.
    """
    # This generates the derived parameters as well as the fundamental parameters
    new_fit_parameters = model.make_params()
    # We then overwrite some of the params to add a good initial guess.
    for index, maximum in enumerate(maxima_params):
        prefix = f"maximum_{index}"
        # If the params have been passed on then use them
        if old_fit_params and f"{prefix}_center" in old_fit_params:
            new_fit_parameters[f"{prefix}_center"] = old_fit_params[f"{prefix}_center"]
            new_fit_parameters[f"{prefix}_sigma"] = old_fit_params[f"{prefix}_sigma"]
            new_fit_parameters[f"{prefix}_fraction"] = old_fit_params[f"{prefix}_fraction"]
            new_fit_parameters[f"{prefix}_amplitude"] = old_fit_params[f"{prefix}_amplitude"]
        # If params haven't been passed on then guess new ones
        else:
            maximum_mask = np.logical_and(x_data > maximum.bounds[0], x_data < maximum.bounds[1])
            maxima_x = x_data[maximum_mask]
            maxima_y = y_data[maximum_mask]
            center = maxima_x[np.argmax(maxima_y)]

            max_sigma, min_sigma, sigma = guess_sigma(x_data, maximum.bounds)
            # When calculating amplitude take the maximum height of the peak but the minimum height
            # of the dataset overall. This is because the maximum_mask does not necessarily
            # include baseline points and we need the noise level.
            amplitude = (max(maxima_y) - min(y_data)) * 2 * sigma
            new_fit_parameters.add(f"{prefix}_center", value=center, min=maximum.bounds[0],
                                   max=maximum.bounds[1])
            new_fit_parameters.add(f"{prefix}_sigma", value=sigma, min=min_sigma, max=max_sigma)
            new_fit_parameters.add(f"{prefix}_fraction", value=0.2, min=0, max=1)
            new_fit_parameters.add(f"{prefix}_amplitude", value=amplitude, min=0)

    if old_fit_params and "background" in old_fit_params:
        new_fit_parameters["background"] = old_fit_params["background"]
    else:
        # Background should be > 0, but a little flexibility here improves fit convergence.
        new_fit_parameters.add("background", value=min(y_data), min=-10, max=max(y_data))
    return new_fit_parameters


def guess_sigma(x_data: np.ndarray,
                maximum_range: Tuple[float, float]) -> Tuple[float, float, float]:
    """Guess an initial value of sigma for the Pseudo-Voigt fit.

    :param x_data: The x_data to be fitted.
    :param maximum_range: Two floats indicating the range of values that the maximum falls within.
    :return: A maximum possible value for sigma, a minimum possible value and the initial guess
      of sigma.
    """

    # By definition in the PV fit, sigma is half the width of the peak at FHWM.
    # In the case of a single peak, the maximum range is set to the peak bounds
    # In the case of multiplet peaks the maximum range is set approximately at the
    # FWHM either side of the peak.
    x_range = max(x_data) - min(x_data)
    maximum_range = maximum_range[1] - maximum_range[0]

    if maximum_range > 0.8 * x_range:
        # If the maximum range is similar to the x_range then we have a single peak. Make
        # assumptions based on data width
        # Sigma is approximately 7% of the peak_bounds
        sigma = 0.07 * x_range
        # The minimum sigma is approximately 2.5% of the peak bounds
        min_sigma = 0.025 * x_range
        # The maximum sigma is approximately 20% of the peak bounds
        max_sigma = 0.20 * x_range

    else:
        # We are dealing with multiple peaks - set sigma to be close to the maxima range
        sigma = 0.5 * maximum_range
        min_sigma = 0.1 * maximum_range
        max_sigma = 4 * maximum_range

    return max_sigma, min_sigma, sigma
