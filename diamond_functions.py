import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import PseudoVoigtModel
from lmfit import Model
from typing import List
import time
import os


def calc_dspacing(ttheta):
    """ Calculate d-spacing from two-theta values.
    """ 
    x_ray_energy = 89.07 # in keV
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
    strain = -(theta-theta0)/np.tan(theta)
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


# Functions for loading up data and fitting
def get_cake(file_path, cake=1):
    """ Return 'spectrum' containing 2-theta increments and intensity values for a given cake.
        Note, assumed DAWN output data has 2-theta in column 0 and intensity of first cake in column 1.
    """
    spectrum = np.loadtxt(file_path, usecols=(0, cake))
    return spectrum
              

def get_spectrum_subset(spectrum, ttheta_lims=(0, 10)):
    """ Return intensity values within a given 2-theta range for an individual lattice plane peak.
        Note, output 'peak' includes 2-theta increments in column 0 and intensity in column 1.
    """
    mask = np.logical_and(spectrum[:, 0] > ttheta_lims[0], spectrum[:, 0] < ttheta_lims[1])
    return spectrum[mask]


def line(x, constBG):
        "constant Background"
        return constBG


def fit_peak(peak_data, initParams=None):
    """ Pseudo-Voigt fit to the lattice plane peak intensity.
        Return results of the fit as an lmfit class, which contains the fitted parameters (amplitude, fwhm, etc.) 
        and the fit line calculated using the fit parameters and 100x two-theta points.
    """
    ttheta = peak_data[:, 0]
    intensity = peak_data[:, 1]
    pvModel = PseudoVoigtModel()
    model = pvModel + Model(line)
    if initParams is None:
        pars = pvModel.guess(intensity, x=ttheta)
        pars['sigma'].set(min=0.01, max=0.02)
        pars['amplitude'].set(min=0.08)
        pars.add("constBG", 0)
    else:
        pars = initParams
    fit_results = model.fit(intensity, pars, x=ttheta)
    fit_ttheta = np.linspace(ttheta[0], ttheta[-1], 100)
    fit_line = [fit_ttheta, model.eval(fit_results.params, x=fit_ttheta)]
    return fit_results, fit_line


# Functions for fitting two and three peaks
def fit_two_peaks(peak_data, pv_1_cent=6.45, pv_1_min=6.37, pv_1_max=6.47, pv_2_cent=6.54, pv_2_min=6.46, pv_2_max=6.56, initParams=None):
    """ Pseudo-Voigt fit to the lattice plane peak intensity for two peaks (adjust function to include more).
        Parameters for each peak also require limits and centres to be set in the case of overlapping peaks.
        Return results of the fit as an lmfit class, which contains the fitted parameters (amplitude, fwhm, etc.) 
        and the fit line calculated using the fit parameters and 100x two-theta points.
    """
    ttheta = peak_data[:, 0]
    intensity = peak_data[:, 1]

    PV_1 = PseudoVoigtModel(prefix='pv_1')
    PV_2 = PseudoVoigtModel(prefix='pv_2')

    model = PV_1 + PV_2 + Model(line)
    
    if initParams is None:      
#if reflection == '(11-22),(20-21)': use starting centres 6.45 (min=6.37,max=6.47), 6.54 (min=6.46,max=6.56)...
#if reflection == '(0004),(220)': use starting centres 6.78 (min=6.37,max=6.85), 6.94(min=6.86,max=7.02)...
        pars_1 = PV_1.guess(intensity, x=ttheta)
        #note, min and max values are not the same as bounds around the peak, but values that they can go up to!
        pars_1['pv_1center'].set(pv_1_cent, min=pv_1_min, max=pv_1_max)
        pars_1['pv_1sigma'].set(min=0.01, max=0.02)
        pars_1['pv_1amplitude'].set(min=0.05)

        pars_2 = PV_2.guess(intensity, x=ttheta)
        pars_2['pv_2center'].set(pv_2_cent,min=pv_2_min, max=pv_2_max)
        pars_2['pv_2sigma'].set(min=0.01, max=0.02)
        pars_2['pv_2amplitude'].set(min=0.05)

        pars=pars_1 + pars_2
        pars.add("constBG", 0)
            
    else:
        pars = initParams
    fit_results = model.fit(intensity, pars, x=ttheta)
    fit_ttheta=np.linspace(ttheta[0],ttheta[-1],100)
    fit_line = [fit_ttheta,model.eval(fit_results.params,x=fit_ttheta)]
    return fit_results, fit_line


def fit_three_peaks(peak_data, pv_1_cent=3.43, pv_1_min=3.36, pv_1_max=3.44, pv_2_cent=3.52, pv_2_min=3.42, pv_2_max=3.55, pv_3_cent=3.59, pv_3_min=3.54, pv_3_max=3.6, initParams=None):
    """ Pseudo-Voigt fit to the lattice plane peak intensity for three peaks (adjust function to include more).
        Parameters for each peak also require limits and centres to be set in the case of overlapping peaks.
        Return results of the fit as an lmfit class, which contains the fitted parameters (amplitude, fwhm, etc.) 
        and the fit line calculated using the fit parameters and 100x two-theta points.
    """
    ttheta = peak_data[:, 0]
    intensity = peak_data[:, 1]

    PV_1 = PseudoVoigtModel(prefix='pv_1')
    PV_2 = PseudoVoigtModel(prefix='pv_2')
    PV_3 = PseudoVoigtModel(prefix='pv_3')

    model = PV_1 + PV_2 + PV_3 + Model(line)
    
    if initParams is None:
# if reflection == '(0002),(110),(10-11)': use starting centres 3.43(min=3.36,max=3.44), 3.54(min=3.42,max=3.55), 3.59(min=3.54,max=3.6)...
#if reflection == '(20-20),(11-22),(20-21)': use starting centres 6.32(min=6.23,max=6.33), 6.45(min=6.37,max=6.47), 6.54(min=6.46,max=6.56)...
        pars_1 = PV_1.guess(intensity, x=ttheta)
        #note, min and max values are not the same as bounds around the peak, but values that they can go up to!
        pars_1['pv_1center'].set(pv_1_cent,min=pv_1_min, max=pv_1_max)
        pars_1['pv_1sigma'].set(min=0.01, max=0.02)

        pars_2 = PV_2.guess(intensity, x=ttheta)
        pars_2['pv_2center'].set(pv_2_cent, min=pv_2_min, max=pv_2_max)
        pars_2['pv_2sigma'].set(min=0.01, max=0.02)

        pars_3 = PV_3.guess(intensity, x=ttheta)
        pars_3['pv_3center'].set(pv_3_cent,min=pv_3_min, max=pv_3_max)
        pars_3['pv_3sigma'].set(min=0.01, max=0.02)

        pars=pars_1 + pars_2 + pars_3
        pars.add("constBG", 0)

    else:
        pars = initParams
    fit_results = model.fit(intensity, pars, x=ttheta)
    fit_ttheta = np.linspace(ttheta[0], ttheta[-1], 100)
    fit_line = [fit_ttheta, model.eval(fit_results.params, x=fit_ttheta)]
    return fit_results, fit_line


class FitCake:
    def __init__(self, file_path, cake):
        self.data_dict = {}
        self.fits_dict = {}
        self.lines_dict = {}
        self.spectrum = get_cake(file_path, cake=cake)
        self.reflection_list = []

    def plot_fit(self, reflection):
        """ Plot the line fit and intensity measurements.
        Input peak labels i.e. (10-10), (0002), etc.
        """
        plt.figure(figsize=(10, 8))
        plt.minorticks_on()
        plt.plot(self.lines_dict[reflection][:, 0], self.lines_dict[reflection][:, 1], linewidth=3)
        plt.plot(self.data_dict[reflection][:, 0], self.data_dict[reflection][:, 1], '+', markersize=15, mew=3)
        plt.xlabel(r'Two Theta ($^\circ$)', fontsize=28)
        # why r? - so can print out latex without needing double slash
        plt.title(reflection, fontsize=28)
        plt.ylabel('Intensity', fontsize=28)
        plt.tight_layout()

    def plot_spectrum(self, xmin=0, xmax=10):
        """Plot the intensity spectrum."""
        plt.figure(figsize=(10, 8))
        plt.minorticks_on()
        plt.plot(self.spectrum[:, 0], self.spectrum[:, 1], '-', linewidth=3)
        plt.xlabel(r'Two Theta ($^\circ$)', fontsize=28)
        plt.ylabel('Intensity', fontsize=28)
        plt.xlim(xmin, xmax)
        plt.tight_layout()


class FitSingleCake(FitCake):
    """ Class for reading in individual cakes and fitting multiple single peaks
        See examples below for usage.
    """

    def __init__(self, file_path, cake):
        super().__init__(file_path, cake)

    def fit_peaks(self, reflection_list: List[str], peak_ranges: List[tuple], init_params=None):
        """ Attempt to fit peaks within the ranges specified by `peak_ranges`.
        :param reflection_list: One label for each peak.
        :param peak_ranges: A tuple for each peak specifying where on the x-axis the peak begins
        and ends.
        :param init_params: Initial parameters to the fitting function
        """
        self.reflection_list = reflection_list
        # zip iterates through each list together
        for reflection, p_range in zip(reflection_list, peak_ranges):
            # store data in dictionary with peak label as the key
            self.data_dict[reflection] = get_spectrum_subset(self.spectrum, ttheta_lims=p_range)

            if init_params:
                fit_results, fit_line = fit_peak(self.data_dict[reflection],
                                                 initParams=init_params.fits_dict[reflection].params)
            else:
                fit_results, fit_line = fit_peak(self.data_dict[reflection])
            self.fits_dict[reflection] = fit_results
            # Transpose the array to get appropriate row/column order.
            self.lines_dict[reflection] = np.array(fit_line).T


class Fit2Peak(FitCake):
    """ Class for reading in individual cakes and fitting two peaks at the same time
    """
    def __init__(self, file_path, cake):
        super().__init__(file_path, cake)

    def fit_2_peaks(self, reflection_list, peak_ranges, pv_1_cent, pv_1_min, pv_1_max, pv_2_cent, pv_2_min, pv_2_max,
                    init_params=None):

        """ Input reflection the collection of peak labels i.e. (0004),(220) and a 2-theta range enclosing the peaks.
            Calculate results of the fit (amplitude, fwhm, etc.) and the fit line and store within the dictionaries.
        """
        self.reflection_list = reflection_list
        # zip iterates through each list together
        for reflection, p_range in zip(reflection_list, peak_ranges):
            peak_data = get_spectrum_subset(self.spectrum, ttheta_lims=p_range)
            self.data_dict[reflection] = peak_data
            # store data in dictionary with peak label as the key
        for reflection, peak_data in self.data_dict.items():
            # in this case the fit_two_peaks function is needed
            if init_params:
                fit_results, fit_line = fit_two_peaks(peak_data, initParams=init_params.fits_dict[
                    reflection].params)
            else:
                fit_results, fit_line = fit_two_peaks(peak_data, pv_1_cent, pv_1_min, pv_1_max, pv_2_cent, pv_2_min,
                                                  pv_2_max)
            self.fits_dict[reflection] = fit_results
            self.lines_dict[reflection] = np.array(fit_line).T


class Fit3Peak(FitCake):
    """ Class for reading in individual cakes and fitting three peaks at the same time
    """
    def __init__(self, file_path, cake):
        super().__init__(file_path, cake)

    def fit_3_peaks(self, reflection_list, peak_ranges, pv_1_cent, pv_1_min, pv_1_max, pv_2_cent,
                    pv_2_min, pv_2_max, pv_3_cent, pv_3_min, pv_3_max, init_params=None):
        self.reflection_list = reflection_list
        # zip iterates through each list together
        for reflection, p_range in zip(reflection_list, peak_ranges):
            peak_data = get_spectrum_subset(self.spectrum, ttheta_lims=p_range)
            self.data_dict[reflection] = peak_data
            # store data in dictionary with peak label as the key
        for reflection, peak_data in self.data_dict.items():
            # in this case the fit_three_peaks function is needed
            if init_params:
                fit_results, fit_line = fit_three_peaks(peak_data, initParams=init_params.fits_dict[
                    reflection].params)
            else:
                fit_results, fit_line = fit_three_peaks(peak_data, pv_1_cent, pv_1_min, pv_1_max,
                                                        pv_2_cent, pv_2_min, pv_2_max, pv_3_cent,
                                                        pv_3_min, pv_3_max, initParams=None)
            self.fits_dict[reflection] = fit_results
            self.lines_dict[reflection] = np.array(fit_line).T


#### Functions for running through all the 'images' - single peak case ####
def run_thru_images(filePrefix, dirname, firstFile, lastFile, peak_bounds_init, peak_labels_init, caking_type, step=1,
                    cake=1):
    """  Run through all 'images of the caked data and create a FitCake class object for each.
         Return a dictionary called fits contains keys (with the image name/number) and a FitCake class object for each. 
         The FitCake class objects contain;
            - a list of reflections (reflection_list)
            - a dictionary containing data (2-theta and intensity) for each reflection (data_dict)
            - a dictionary containing a fitted line to the data, for each relection (lines_dict)
            - a dictionary containing the class object from the lmfit model for each reflection (fits_dict)
        """
    # record the start time, note import time.
    start_time = time.time()

    fits = dict()

    peak_bounds_copy = peak_bounds_init.copy()
    peak_labels_copy = peak_labels_init.copy()
    # peak_bounds_copy[0]=(3.10,3.30)
    # can't change single element, since peak_bounds_copy[0][0]=3.2 -> assignment error.

    for image_number in range(firstFile, lastFile + 1, step):

        if caking_type == 'normal':
            # dirname='Data/'+filePrefix + '_ascii/'
            fnumber = '_{:05d}'.format(image_number)
            fname = filePrefix + fnumber + '.dat'

        if caking_type == 'bottom' or caking_type == 'top' or caking_type == 'vertical' or caking_type == 'horizontal':
            # dirname='Data/'+filePrefix + '_ascii/Merge/'
            fnumber = '_{:05d}'.format(image_number)
            fname = filePrefix + '_MergeCakePoints_' + caking_type + fnumber + '.dat'

        # create a class instance
        fitted_cake = FitSingleCake(dirname + fname, cake)
        fitted_cake.fit_peaks(peak_labels_copy, peak_bounds_copy)
        fits[filePrefix + fnumber] = fitted_cake

        for i, (bounds, labels) in enumerate(zip(peak_bounds_copy, peak_labels_copy)):
            # Re-centre the peak bounds.
            thetaHalfRange = (bounds[1] - bounds[0]) / 2
            center = fits[filePrefix + fnumber].fits_dict[labels].values['center']
            peak_bounds_copy[i] = (center - thetaHalfRange, center + thetaHalfRange)
            # Can't change single element of a tuple as this returns an assignment error.

        print(image_number)

    # print how long the analysis has taken
    print("--- %s seconds ---" % (time.time() - start_time))

    return fits


# Functions for running through all the 'images' - multiple peak case (passing on initial parameters)
def run_thru_images_initParams(filePrefix, dirname, firstFile, lastFile, peak_bounds_init, peak_labels_init,
                               caking_type, peak_number, pv_1_cent=0, pv_1_min=0, pv_1_max=0, pv_2_cent=0, pv_2_min=0,
                               pv_2_max=0, pv_3_cent=0, pv_3_min=0, pv_3_max=0, step=1, cake=1):
    """  Run through all 'images of the caked data and create a FitCake class object for each.
         Passes on initial parameters to help fit multiple peaks.
         Return a dictionary called fits contains keys (with the image name/number) and a FitCake class object for each. 
         The FitCake class objects contain;
            - a list of reflections (reflection_list)
            - a dictionary containing data (2-theta and intensity) for each reflection (data_dict)
            - a dictionary containing a fitted line to the data, for each relection (lines_dict)
            - a dictionary containing the class object from the lmfit model for each reflection (fits_dict)
        """
    # record the start time, note import time.
    start_time = time.time()

    fits = dict()

    peak_bounds_copy = peak_bounds_init.copy()
    peak_labels_copy = peak_labels_init.copy()
    # peak_bounds_copy[0]=(3.10,3.30)
    # can't change single element, since peak_bounds_copy[0][0]=3.2 -> assignment error.

    firstIter = True

    if firstIter:

        if caking_type == 'normal':
            # dirname='Data/'+filePrefix + '_ascii/'
            fnumber = '_{:05d}'.format(firstFile)
            fname = filePrefix + fnumber + '.dat'

        if caking_type == 'bottom' or caking_type == 'top' or caking_type == 'vertical' or caking_type == 'horizontal':
            # dirname='Data/'+filePrefix + '_ascii/Merge/'
            fnumber = '_{:05d}'.format(firstFile)
            fname = filePrefix + '_MergeCakePoints_' + caking_type + fnumber + '.dat'

        if peak_number == 'one':

            # create a class instance
            fitted_cake = FitSingleCake(dirname + fname, cake)
            fitted_cake.fit_peaks(peak_labels_copy, peak_bounds_copy)
            fits[filePrefix + fnumber] = fitted_cake

            for i, (bounds, labels) in enumerate(zip(peak_bounds_copy, peak_labels_copy)):
                # Re-centre the peak bounds.
                thetaHalfRange = (bounds[1] - bounds[0]) / 2
                center = fits[filePrefix + fnumber].fits_dict[labels].values['center']
                peak_bounds_copy[i] = (center - thetaHalfRange, center + thetaHalfRange)
                # Can't change single element of a tuple as this returns an assignment error.

        if peak_number == 'two':
            # create a class instance
            fitted_cake = Fit2Peak(dirname, fname, cake)
            fitted_cake.fit_2_peaks(peak_labels_copy, peak_bounds_copy, pv_1_cent, pv_1_min, pv_1_max, pv_2_cent,
                                    pv_2_min, pv_2_max)
            fits[filePrefix + fnumber] = fitted_cake

            # no recentering of peak bounds needed

        if peak_number == 'three':
            # create a class instance
            fitted_cake = Fit3Peak(dirname, fname, cake)
            fitted_cake.fit_3_peaks(peak_labels_copy, peak_bounds_copy, pv_1_cent, pv_1_min, pv_1_max, pv_2_cent,
                                    pv_2_min, pv_2_max, pv_3_cent, pv_3_min, pv_3_max)
            fits[filePrefix + fnumber] = fitted_cake

            # no recentering of peak bounds needed

    for image_number in range(firstFile + 1, lastFile + 1, step):

        if caking_type == 'normal':
            # dirname='Data/'+filePrefix + '_ascii/'
            fnumber = '_{:05d}'.format(image_number)
            fnumber_previous = '_{:05d}'.format(image_number - 1)
            fname = filePrefix + fnumber + '.dat'

        if caking_type == 'bottom' or caking_type == 'top' or caking_type == 'vertical' or caking_type == 'horizontal':
            # dirname='Data/'+filePrefix + '_ascii/Merge/'
            fnumber = '_{:05d}'.format(image_number)
            fnumber_previous = '_{:05d}'.format(image_number - 1)
            fname = filePrefix + '_MergeCakePoints_' + caking_type + fnumber + '.dat'

        if peak_number == 'one':

            # create a class instance
            fitted_cake = FitSingleCake(dirname + fname, cake)
            fitted_cake.fit_peaks(peak_labels_copy, peak_bounds_copy,
                                              init_params=fits[filePrefix + fnumber_previous])
            fits[filePrefix + fnumber] = fitted_cake

            for i, (bounds, labels) in enumerate(zip(peak_bounds_copy, peak_labels_copy)):
                # Re-centre the peak bounds.
                thetaHalfRange = (bounds[1] - bounds[0]) / 2
                center = fits[filePrefix + fnumber].fits_dict[labels].values['center']
                peak_bounds_copy[i] = (center - thetaHalfRange, center + thetaHalfRange)
                # Can't change single element of a tuple as this returns an assignment error.

        if peak_number == 'two':
            # create a class instance
            fitted_cake = Fit2Peak(dirname, fname, cake)
            fitted_cake.fit_2_peaks(peak_labels_copy, peak_bounds_copy,
                                    init_params=fits[filePrefix + fnumber_previous])
            fits[filePrefix + fnumber] = fitted_cake

            # no recentering of peak bounds needed

        if peak_number == 'three':
            # create a class instance
            fitted_cake = Fit3Peak(dirname, fname, cake)
            fitted_cake.fit_3_peaks(peak_labels_copy, peak_bounds_copy,
                                    init_params=fits[filePrefix + fnumber_previous])
            fits[filePrefix + fnumber] = fitted_cake

            # no recentering of peak bounds needed

        print(image_number)

    # print how long the analysis has taken
    print("--- %s seconds ---" % (time.time() - start_time))
    return fits


# Functions for plotting saved data ####
def plot_fit_saved_data(ref, line, data):
    """ Plot the line fit and intensity measurements.
        Input peak labels i.e. (10-10), (0002), etc.
    """
    plt.figure(figsize=(10, 8))
    plt.minorticks_on()
    plt.plot(line[:, 0], line[:, 1], linewidth=3)
    plt.plot(data[:, 0], data[:, 1], '+', markersize=15, mew=3)
    plt.xlabel(r'Two Theta ($^\circ$)', fontsize=28)
    plt.title(ref, fontsize=28)
    plt.ylabel('Intensity', fontsize=28)
    plt.tight_layout()


#### Function to create merged cake files for increasing intensity ####
def merge_peak_intensity(filePrefix, dataFolder, fileNameExtension, fileType, firstFile, lastFile, caking_type, step=1):
    """  Create a file merging the peak intensities of the given cakes, increasing peak intensity.
        Options for cakes are 'bottom, 'top', 'vertical', 'horizontal'.
    """
    for image_number in range(firstFile, lastFile + 1, step):

        dirname = dataFolder + filePrefix + fileNameExtension
        fnumber = '_{:05d}'.format(image_number)
        fname = filePrefix + fnumber + fileType
        path = dirname + fname

        if caking_type == 'bottom':
            cake1 = np.loadtxt(path, skiprows=1, usecols=(9))
            # note, column 0 is two-theta values. Column 1 is right hand cake at -5 to 5 deg if using 10 deg slices i.e. in Dawn (-5,355)
            cake2 = np.loadtxt(path, skiprows=1, usecols=(10))
            cake3 = np.loadtxt(path, skiprows=1, usecols=(11))
            sum_cake_intensity = cake1 + cake2 + cake3

        if caking_type == 'top':
            cake1 = np.loadtxt(path, skiprows=1, usecols=(27))
            cake2 = np.loadtxt(path, skiprows=1, usecols=(28))
            cake3 = np.loadtxt(path, skiprows=1, usecols=(29))
            sum_cake_intensity = cake1 + cake2 + cake3

        if caking_type == 'vertical':
            cake1 = np.loadtxt(path, skiprows=1, usecols=(9))
            cake2 = np.loadtxt(path, skiprows=1, usecols=(10))
            cake3 = np.loadtxt(path, skiprows=1, usecols=(11))
            cake4 = np.loadtxt(path, skiprows=1, usecols=(27))
            cake5 = np.loadtxt(path, skiprows=1, usecols=(28))
            cake6 = np.loadtxt(path, skiprows=1, usecols=(29))
            sum_cake_intensity = cake1 + cake2 + cake3 + cake4 + cake5 + cake6

        if caking_type == 'horizontal':
            cake1 = np.loadtxt(path, skiprows=1, usecols=(36))
            cake2 = np.loadtxt(path, skiprows=1, usecols=(1))
            cake3 = np.loadtxt(path, skiprows=1, usecols=(2))
            cake4 = np.loadtxt(path, skiprows=1, usecols=(18))
            cake5 = np.loadtxt(path, skiprows=1, usecols=(19))
            cake6 = np.loadtxt(path, skiprows=1, usecols=(20))
            sum_cake_intensity = cake1 + cake2 + cake3 + cake4 + cake5 + cake6

        ttheta = np.loadtxt(path, skiprows=1, usecols=(0))
        merge = np.array([ttheta, sum_cake_intensity]).T
        # merge=np.stack([ttheta,sum_cake_intensity],axis=1)

        newfilePrefix = filePrefix + "_MergeCakeIntensity_" + caking_type
        newfname = newfilePrefix + fnumber + fileType
        newpath = dirname + 'Merge/' + newfname

        os.makedirs(os.path.dirname(newpath), exist_ok=True)
        np.savetxt(newpath, merge)


#### Function to create merged cake files for greater no. of points ####
def merge_peak_points(filePrefix, dataFolder, fileNameExtension, fileType, firstFile, lastFile, caking_type, step=1):
    """  Create a file merging the given cakes, giving a greater number of points at each 2-theta value.
        Options for cakes are 'bottom, 'top', 'vertical', 'horizontal'.
        """
    for image_number in range(firstFile, lastFile + 1, step):

        dirname = dataFolder + filePrefix + fileNameExtension
        fnumber = '_{:05d}'.format(image_number)
        fname = filePrefix + fnumber + fileType
        path = dirname + fname

        if caking_type == 'bottom':
            cakes = np.loadtxt(path, skiprows=1, usecols=(0, 9, 10, 11))
            # note, column 0 is two-theta values. Column 1 is right hand cake at -5 to 5 deg if using 10 deg slices i.e. in Dawn (-5,355)
            merge = np.empty([3078, 2])  # note, 1026 values of 2-theta

            for i, row in enumerate(cakes):
                merge[3 * i:3 * i + 3, 0] = row[0]
                merge[3 * i, 1] = row[1]
                merge[3 * i + 1, 1] = row[2]
                merge[3 * i + 2, 1] = row[3]

        if caking_type == 'top':
            cakes = np.loadtxt(path, skiprows=1, usecols=(0, 27, 28, 29))
            merge = np.empty([3078, 2])  # note, 1026 values of 2-theta

            for i, row in enumerate(cakes):
                merge[3 * i:3 * i + 3, 0] = row[0]
                merge[3 * i, 1] = row[1]
                merge[3 * i + 1, 1] = row[2]
                merge[3 * i + 2, 1] = row[3]

        if caking_type == 'vertical':
            cakes = np.loadtxt(path, skiprows=1, usecols=(0, 9, 10, 11, 27, 28, 29))
            merge = np.empty([6156, 2])

            for i, row in enumerate(cakes):
                merge[6 * i:6 * i + 6, 0] = row[0]
                merge[6 * i, 1] = row[1]
                merge[6 * i + 1, 1] = row[2]
                merge[6 * i + 2, 1] = row[3]
                merge[6 * i + 3, 1] = row[4]
                merge[6 * i + 4, 1] = row[5]
                merge[6 * i + 5, 1] = row[6]

        if caking_type == 'horizontal':
            cakes = np.loadtxt(path, skiprows=1, usecols=(0, 36, 1, 2, 18, 19, 20))
            merge = np.empty([6156, 2])

            for i, row in enumerate(cakes):
                merge[6 * i:6 * i + 6, 0] = row[0]
                merge[6 * i, 1] = row[1]
                merge[6 * i + 1, 1] = row[2]
                merge[6 * i + 2, 1] = row[3]
                merge[6 * i + 3, 1] = row[4]
                merge[6 * i + 4, 1] = row[5]
                merge[6 * i + 5, 1] = row[6]

        newfilePrefix = filePrefix + "_MergeCakePoints_" + caking_type
        newfname = newfilePrefix + fnumber + fileType
        newpath = dirname + 'Merge/' + newfname

        os.makedirs(os.path.dirname(newpath), exist_ok=True)

        np.savetxt(newpath, merge)