from xrdfit.spectrum_fitting import PeakParams, FitSpectrum, FittingExperiment

frame_time = 1
file_string = '../example_data/example_data_large/adc_065_TI64_NDload_900C_15mms_{:05d}.dat'
first_cake_angle = 90
cakes_to_fit = 1

peak_params = [PeakParams('(10-10)', (3.02, 3.27)),
               PeakParams('(0002)(110)(10-11)',  (3.3, 3.75), [(3.4, 3.44), (3.52, 3.56), (3.57, 3.61)]),
               PeakParams('(10-12)', (4.54, 4.8)),
               PeakParams('(200)', (4.9, 5.10)),
               PeakParams('(11-20)', (5.35, 5.6)),
               PeakParams('(10-13)', (5.9, 6.15), [(6.00, 6.05)]),
               PeakParams('(20-20)', (6.21, 6.4)),
               PeakParams('(11-22)(20-21)',  (6.37, 6.71), [(6.43, 6.47), (6.52, 6.56)]),
               PeakParams('(0004)',  (6.75, 6.95), [(6.82, 6.87)]),
               PeakParams('(220)(20-22)', (6.95, 7.35), [(7.05, 7.12), (7.16, 7.20)]),
               PeakParams('(310)', (7.75, 8.05))
              ]
max_frame = 5657
merge_cakes = False
frames_to_fit = range(1, max_frame, 100)
experiment = FittingExperiment(frame_time, file_string, first_cake_angle, cakes_to_fit, peak_params, merge_cakes, frames_to_fit)

experiment.run_analysis(reuse_fits=True)