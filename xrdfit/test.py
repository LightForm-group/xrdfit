import xrdfit.spectrum_fitting as spectrum_fitting
from xrdfit.spectrum_fitting import PeakParams


frame_time = 10
file_stub = "../example_data/adc_041_7Nb_NDload_700C_15mms_*"
first_cake_angle = 90
cakes_to_fit = [36, 1, 2]
peak_params = PeakParams('1', (2.8, 2.9))
merge_cakes = True

experiment = spectrum_fitting.FittingExperiment(frame_time, file_stub, first_cake_angle,
                                                cakes_to_fit, peak_params, merge_cakes)

experiment.run_analysis()
for peak_name in experiment.peak_names():
    for parameter in experiment.fit_parameters(peak_name):
        experiment.plot_fit_parameter(peak_name, parameter)