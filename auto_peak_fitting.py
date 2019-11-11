import spectrum_fitting
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

file_path = 'example_data/adc_041_7Nb_NDload_700C_15mms_00001.dat'
spectral_data = spectrum_fitting.FitSpectrum(file_path)

sub_spectrum = spectral_data.get_spectrum_subset(1, (2, 8.5), merge_cakes=True)
peaks, peak_properties = find_peaks(sub_spectrum[:, 1], height=[None, None],
                                    width=[2, None], prominence=[2, None])
plt.plot(sub_spectrum[:, 0], sub_spectrum[:, 1])
for peak in peaks:
    plt.axvline(x=sub_spectrum[peak, 0], color="green")
plt.show()
