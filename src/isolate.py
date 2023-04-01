import os

import pandas as pd
import numpy as np

from scipy.io import wavfile
from scipy import signal, interpolate
from matplotlib import pyplot as plt
from util import *


def isolate(label, input_path, output_path = 'output.wav'):
    # Read the audio file
    data_dir = "training_data"
    
    input_sample_rate, input_audio = wavfile.read(input_path)
    input_audio = input_audio[:, 0]

    # Get the csv data
    inst_freqs_all = np.array(pd.read_csv("inst_freqs.csv"))
    # Get the frequencies for the selected instrument and remove label
    inst_freqs = inst_freqs_all[np.where(inst_freqs_all[:, 0] == label)]
    inst_freqs = inst_freqs[0, 1:]
    exp_bins = get_exp_bins()

    filtered = []

    # This defines how wide our band pass filters will be
    # Should consider making all of the audio processing on an exponential scale
    bin_width = 4 * SAMPLE_RATE / SAMPLE_SIZE
    for i in range(len(inst_freqs)):
        bin = exp_bins[i]
        freq = inst_freqs[i]
        # bounds for filter where lower bound is halfway between current bin and previous bin and 
        # upper bound is halfway between current bin and next bin on linear scale, when translated
        # to exponential scale this is the calculation
        lb = bin * pow(2, -0.01689 / 2)
        ub = bin * pow(2, 0.01689 / 2)

        # Low pass if the lowerbound is 0 or less
        if lb <= 0:
            sos = signal.butter(5, ub, 'lowpass', fs=SAMPLE_RATE, output='sos')
        # High pass if the upperbound is greater than the Nyquist frequency
        elif ub >= SAMPLE_RATE / 2:
            sos = signal.butter(5, lb, 'highpass', fs=SAMPLE_RATE, output='sos')
        # Band pass for all other cases
        else:
            sos = signal.butter(5, [lb, ub], 'bandpass', fs=SAMPLE_RATE, output='sos')
        
        # Apply the filter and multiply by the weight
        new_audio = signal.sosfiltfilt(sos, input_audio).astype(np.int16) * freq
        filtered.append(new_audio)

    # Average all of the filters applied
    final = np.average(filtered, axis=0)
    # Boost audio to a reasonable level and write to output
    final = 0.5 * final * np.iinfo(np.int16).max / np.max(final)
    wavfile.write(output_path, SAMPLE_RATE, final.astype(np.int16))
    

def main():
    label = 'tru'
    data_dir = 'training_data'
    input_file_path = os.path.join(data_dir, label, '[tru][cla]1870__1.wav')
    isolate(label, input_file_path)

if __name__ == '__main__':
    main()