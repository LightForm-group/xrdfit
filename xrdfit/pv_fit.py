from typing import List, Tuple

import lmfit
import numpy as np


def do_pv_fit(peak_data: np.ndarray, maxima_locations: List[Tuple[float, float]],
              fit_parameters: lmfit.Parameters = None):
    """
    Pseudo-Voigt fit to the lattice plane peak intensity.
    Return results of the fit as an lmfit class, which contains the fitted parameters
    (amplitude, fwhm, etc.) and the fit line calculated using the fit parameters and
    100x two-theta points.
    """
    model = None

    num_maxima = len(maxima_locations)

    for maxima_num in range(num_maxima):
        # Add the peak to the model
        prefix = f"maximum_{maxima_num + 1}_"
        if model:
            model += lmfit.models.PseudoVoigtModel(prefix=prefix)
        else:
            model = lmfit.models.PseudoVoigtModel(prefix=prefix)

    model += lmfit.Model(lambda background: background)

    two_theta = peak_data[:, 0]
    intensity = peak_data[:, 1]

    if not fit_parameters:
        fit_parameters = model.make_params()
        fit_parameters = guess_params(fit_parameters, two_theta, intensity, maxima_locations)

    fit_result = model.fit(intensity, fit_parameters, x=two_theta, iter_cb=iteration_callback)

    return fit_result


def guess_params(params: lmfit.Parameters, x_data, y_data,
                 maxima_ranges: List[Tuple[float, float]]) -> lmfit.Parameters:
    """Given a dataset and some details about where the maxima are, guess some good initial
    values for the PV fit."""

    for index, maximum in enumerate(maxima_ranges):
        maximum_mask = np.logical_and(x_data > maximum[0],
                                      x_data < maximum[1])
        maxima_x = x_data[maximum_mask]
        maxima_y = y_data[maximum_mask]
        center = maxima_x[np.argmax(maxima_y)]

        max_sigma, min_sigma, sigma = guess_sigma(x_data, maximum)
        # Take the maximum height of the peak but the minimum height of the dataset overall
        # This is because the maximum_mask does not necessarily include baseline points.
        amplitude = (max(maxima_y) - min(y_data)) * 2 * sigma
        param_prefix = f"maximum_{index + 1}"
        params.add(f"{param_prefix}_center", value=center, min=maximum[0], max=maximum[1])
        params.add(f"{param_prefix}_sigma", value=sigma, min=min_sigma, max=max_sigma)
        params.add(f"{param_prefix}_fraction", value=0.2, min=0, max=1)
        params.add(f"{param_prefix}_amplitude", value=amplitude, min=0)
    # Background should be > 0 but a little flexibility here improves the convergence of the fit.
    params.add("background", value=min(y_data), min=-10, max=max(y_data))
    return params


def guess_sigma(x_data, maximum_range):
    # Sigma is half the width of the peak at FHWM
    x_range = max(x_data) - min(x_data)
    maximum_range = maximum_range[1] - maximum_range[0]

    if maximum_range > 0.8 * x_range:
        # If the maximum range is similar to the x_range then we have a single peak. Make
        # assumptions based on data width
        # Sigma is very approximately 7% of the peak_bounds.
        sigma = 0.07 * x_range
        # The minimum sigma is very approximately 2.5% of the peak bounds
        min_sigma = 0.025 * x_range
        # The maximum sigma is very approximately 20% of the peak bounds
        max_sigma = 0.20 * x_range

    else:
        # We are dealing with multiple peaks - set sigma to be close to the maxima range
        sigma = 0.5 * maximum_range
        min_sigma = 0.1 * maximum_range
        max_sigma = 4 * maximum_range

    return max_sigma, min_sigma, sigma


# noinspection PyUnusedLocal
def iteration_callback(parameters, iteration_num, residuals, *args, **kws):
    """This method is called on every iteration of the minimisation. This can be used
    to monitor progress."""
    return False
