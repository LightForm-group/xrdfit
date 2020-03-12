""" This module contain functions implementing the Pseudo-Voigt fit in lmfit.
None of these functions should be called directly by users - these functions are called from
methods in spectrum_fitting.
"""

from typing import List, Tuple, TYPE_CHECKING

import lmfit
import numpy as np

if TYPE_CHECKING:
    from spectrum_fitting import PeakParams


def do_pv_fit(peak_data: np.ndarray, peak_param: "PeakParams") -> lmfit.model.ModelResult:
    """Pseudo-Voigt fit to the lattice plane peak intensity.

    :param peak_data: The data to be fitted, two theta values (x-data) in column 0 and intensity
      (y-data) in column 1.
    :param peak_param: A PeakParams object describing the peak to be fitted.
    """
    model = None
    fit_parameters = peak_param.previous_fit_parameters
    num_maxima = len(peak_param.maxima_bounds)

    # Add one peak to the model for each maximum
    for maxima_num in range(num_maxima):
        prefix = f"maximum_{maxima_num + 1}_"
        if model:
            model += lmfit.models.PseudoVoigtModel(prefix=prefix)
        else:
            model = lmfit.models.PseudoVoigtModel(prefix=prefix)
        model.set_param_hint(f"{prefix}snr", expr=f"{prefix}height/background")
    model += lmfit.Model(lambda background: background)

    two_theta = peak_data[:, 0]
    intensity = peak_data[:, 1]

    if not fit_parameters:
        fit_parameters = model.make_params()
        fit_parameters = guess_params(fit_parameters, two_theta, intensity, peak_param.maxima_bounds)
        # We can't use special characters in param names so have to save the user provided
        # name in user_data.
        for parameter in fit_parameters:
            if parameter != "background":
                parameter_num = int(parameter.split("_")[1])
                fit_parameters[parameter].user_data = peak_param.maxima_names[parameter_num - 1]

    fit_result = model.fit(intensity, fit_parameters, x=two_theta)

    return fit_result


def guess_params(params: lmfit.Parameters, x_data: np.ndarray, y_data: np.ndarray,
                 maxima_ranges: List[Tuple[float, float]]) -> lmfit.Parameters:
    """Given a dataset and some information about where the maxima are, guess some good initial
    values for the Pseudo-Voigt fit.

    :param params: The lmfit.Parameters instance to store the guessed parameters.
    :param x_data: The x data to be fitted.
    :param y_data: The y data to be fitted.
    :param maxima_ranges: A pair of floats for each maximum indicating a range of x-values that
      the maximum falls in.
    """
    for index, maximum in enumerate(maxima_ranges):
        maximum_mask = np.logical_and(x_data > maximum[0],
                                      x_data < maximum[1])
        maxima_x = x_data[maximum_mask]
        maxima_y = y_data[maximum_mask]
        center = maxima_x[np.argmax(maxima_y)]

        max_sigma, min_sigma, sigma = guess_sigma(x_data, maximum)
        # When calculating amplitude take the maximum height of the peak but the minimum height
        # of the dataset overall. This is because the maximum_mask does not necessarily
        # include baseline points and we need the noise level.
        amplitude = (max(maxima_y) - min(y_data)) * 2 * sigma
        param_prefix = f"maximum_{index + 1}"
        params.add(f"{param_prefix}_center", value=center, min=maximum[0], max=maximum[1])
        params.add(f"{param_prefix}_sigma", value=sigma, min=min_sigma, max=max_sigma)
        params.add(f"{param_prefix}_fraction", value=0.2, min=0, max=1)
        params.add(f"{param_prefix}_amplitude", value=amplitude, min=0)
    # Background should be > 0 but a little flexibility here improves the convergence of the fit.
    params.add("background", value=min(y_data), min=-10, max=max(y_data))
    return params


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
