from dataclasses import dataclass

import spectrum_fitting
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

file_path = 'example_data/adc_041_7Nb_NDload_700C_15mms_00001.dat'
spectral_data = spectrum_fitting.FitSpectrum(file_path, cake=1)

sub_spectrum = spectrum_fitting.get_spectrum_subset(spectral_data.spectrum, (2, 8.5))
peaks, peak_properties = find_peaks(sub_spectrum[:, 1], height=[None, None],
                                    width=[1, None], prominence=[5, None])
plt.plot(sub_spectrum[:, 0], sub_spectrum[:, 1])
plt.plot(sub_spectrum[peaks, 0], sub_spectrum[peaks, 1], "x")
#plt.vlines(x=sub_spectrum[peaks, 0], ymin=sub_spectrum[peaks, 1] - peak_properties["prominences"], ymax=sub_spectrum[peaks, 1], color = "C1")
#plt.hlines(y=peak_properties["width_heights"], xmin=peak_properties["left_ips"], xmax = peak_properties["right_ips"], color = "C1")
plt.show()