import os

import pandas as pd
import numpy as np

from scipy.io import wavfile
from scipy import signal
from util import *


# Read the audio file
data_dir = "training_data"
input_file_path = os.path.join(data_dir, "tru", "[tru][cla]1870__1.wav")
input_sample_rate, input_audio = wavfile.read(input_file_path)
input_audio = input_audio[:, 0]


# Instrument to isolate
isolate = "tru"

# Get the csv data
inst_freqs_all = np.array(pd.read_csv("inst_freqs.csv"))
# Get the frequencies for the selected instrument and remove label
inst_freqs = inst_freqs_all[np.where(inst_freqs_all[:, 0] == isolate)]
inst_freqs = inst_freqs[0, 1:]
inst_freqs = list(zip(get_bins(), inst_freqs))

filtered = []

# This defines how wide our band pass filters will be
# Should consider making all of the audio processing on an exponential scale
bin_width = 4 * SAMPLE_RATE / SAMPLE_SIZE
for i in range(len(inst_freqs)):
    bin, freq = inst_freqs[i]
    # bounds for filter
    lb = int(bin - (bin_width / 2))
    ub = int(bin + (bin_width / 2))

    # Low pass if the lowerbound is 0 or less
    if lb <= 0:
        sos = signal.butter(10, ub, 'lowpass', fs=SAMPLE_RATE, output='sos')
    # High pass if the upperbound is greater than the Nyquist frequency
    elif ub >= SAMPLE_RATE / 2:
        sos = signal.butter(10, lb, 'highpass', fs=SAMPLE_RATE, output='sos')
    # Band pass for all other cases
    else:
        sos = signal.butter(10, [lb, ub], 'bandpass', fs=SAMPLE_RATE, output='sos')
    
    # Apply the filter and multiply by the weight
    new_audio = signal.sosfiltfilt(sos, input_audio).astype(np.int16) * freq
    filtered.append(new_audio)

# Average all of the filters applied
final = np.average(filtered, axis=0)
# Boost audio to a reasonable level and write to output
final = 0.5 * final * np.iinfo(np.int16).max / np.max(final)
wavfile.write('test.wav', SAMPLE_RATE, final.astype(np.int16))
    