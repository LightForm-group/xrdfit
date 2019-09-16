from diamond_functions import MaximumParams, FitSingletPeak, FitDoubletPeak, FitTripletPeak
import time
import matplotlib.pyplot as plt
import numpy as np
import os

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

        elif caking_type == 'bottom' or caking_type == 'top' or caking_type == 'vertical' or caking_type == 'horizontal':
            # dirname='Data/'+filePrefix + '_ascii/Merge/'
            fnumber = '_{:05d}'.format(image_number)
            fname = filePrefix + '_MergeCakePoints_' + caking_type + fnumber + '.dat'

        else:
            raise TypeError("Caking type not recognised.")

        # create a class instance
        fitted_cake = FitSingletPeak(dirname + fname, cake)
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
                               caking_type, peak_number, p1: MaximumParams, p2: MaximumParams, p3: MaximumParams, step=1, cake=1):
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

        else:
            raise TypeError("Unknown caking type.")

        if peak_number == 'one':

            # create a class instance
            fitted_cake = FitSingletPeak(dirname + fname, cake)
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
            fitted_cake = FitDoubletPeak(dirname, fname, cake)
            fitted_cake.fit_2_peaks(peak_labels_copy, peak_bounds_copy, p1, p2)
            fits[filePrefix + fnumber] = fitted_cake

            # no recentering of peak bounds needed

        if peak_number == 'three':
            # create a class instance
            fitted_cake = FitTripletPeak(dirname, fname, cake)
            fitted_cake.fit_3_peaks(peak_labels_copy, peak_bounds_copy, p1, p2, p3)
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
            fitted_cake = FitSingletPeak(dirname + fname, cake)
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
            fitted_cake = FitDoubletPeak(dirname, fname, cake)
            fitted_cake.fit_2_peaks(peak_labels_copy, peak_bounds_copy,
                                    init_params=fits[filePrefix + fnumber_previous])
            fits[filePrefix + fnumber] = fitted_cake

            # no recentering of peak bounds needed

        if peak_number == 'three':
            # create a class instance
            fitted_cake = FitTripletPeak(dirname, fname, cake)
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