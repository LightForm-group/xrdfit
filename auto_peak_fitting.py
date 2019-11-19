import plotting
import spectrum_fitting
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np


def detect_peaks():
    for i in range(1, 3401, 200):
        file_path = 'example_data/adc_041_7Nb/adc_041_7Nb_NDload_700C_15mms_{:05d}.dat'.format(i)
        spectral_data = spectrum_fitting.FitSpectrum(file_path)

        sub_spectrum = spectral_data.get_spectrum_subset(1, (2, 8.5), merge_cakes=True)
        peaks, peak_properties = find_peaks(sub_spectrum[:, 1], height=[None, None],
                                            prominence=[2, None], width=[1, None])

        plotting.plot_spectrum(sub_spectrum, [1], False, False, (2.5, 4.5))
        plt.savefig(f"{i:05d}.png", dpi=150)

        doublet_x_threshold = 15
        doublet_y_threshold = 3
        noise_level = np.percentile(sub_spectrum[:, 1], 20)
        non_singlet_peaks = []

        # Identify non-singlet peaks
        for peak_num, peak_index in enumerate(peaks):
            if peak_num + 1 < len(peaks):
                next_peak_index = peaks[peak_num + 1]
                if (next_peak_index - peak_index) < doublet_x_threshold:
                    if np.min(sub_spectrum[peak_index:next_peak_index, 1]) > doublet_y_threshold * noise_level:
                        non_singlet_peaks.append(peak_num)
                        non_singlet_peaks.append(peak_num + 1)

        # Build up list of PeakParams
        peak_params = []
        conversion_factor = sub_spectrum[1, 0] - sub_spectrum[0, 0]
        spectrum_offset = sub_spectrum[0, 0]
        for peak_num, peak_index in enumerate(peaks):
            if peak_num not in non_singlet_peaks:
                left = np.floor(peak_index - 2 * peak_properties["widths"][peak_num]) * conversion_factor + spectrum_offset
                right = np.ceil(peak_index + 2 * peak_properties["widths"][peak_num]) * conversion_factor + spectrum_offset
                peak_params.append(spectrum_fitting.PeakParams(str(peak_num), (left, right)))

        plotting.plot_peak_params(peak_params, x_range=(2.5, 4.5))
        plt.savefig(f"{i:05d}_peaks.png", dpi=150)

        # # Simple peak diagram
        # for peak_num, peak_index in enumerate(peaks):
        #     peak_color = "green"
        #     if peak_num in non_singlet_peaks:
        #         peak_color = "red"
        #     plt.axvline(x=sub_spectrum[peak_index, 0], color=peak_color)
        # plt.savefig(f"{i:05d}_peaks.png", dpi=300)
        # plt.clf()


detect_peaks()
