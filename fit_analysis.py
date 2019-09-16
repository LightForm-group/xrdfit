"""Functions to analyse fitted data."""

import numpy as np


def calc_d_spacing(ttheta):
    """ Calculate d-spacing from two-theta values.
    """
    x_ray_energy = 89.07  # in keV
    c = 2.99792458e8
    h = 6.62607004e-34
    e = 1.6021766208e-19
    x_ray_wavelength = (h * c) / (x_ray_energy * 1e3 * e)

    return x_ray_wavelength / (2 * np.sin(np.array(ttheta) * np.pi / 360))


def calc_strain(ttheta):
    """Calculate strain from two-theta values. Applies average of first 200 points to define
    zero two-theta."""
    theta = 0.5 * (np.array(ttheta)) * np.pi / 180.0
    theta0 = np.mean(theta[0:200])
    strain = -(theta - theta0) / np.tan(theta)
    return strain


def calc_strain_singlepoint(ttheta):
    """Calculate strain from two-theta values. First two-theta values is defined as zero two-theta.
    """
    theta = 0.5 * (np.array(ttheta)) * np.pi / 180.0
    theta0 = theta[0]
    strain = -(theta - theta0) / np.tan(theta)
    return strain


def relative_amplitude(amp):
    """ Calculate difference in amplitude from first measurement."""
    amp0 = amp[2]
    rel_amp = np.array(amp) / amp0
    return rel_amp
