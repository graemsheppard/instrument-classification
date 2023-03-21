import os
import re

import numpy as np

from scipy.io import wavfile
from scipy.fft import fft
from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt


# WAV file values are 16-bit ranging from -32,768 to 32,767
# Sampling rate usually 44,100Hz
# So the max frequency bin of the FFT is sample_rate / 2 = 22,050
# The number of (useful) bins, n, is dependant on sample_size input into FFT divided by 2,
# the other half are symmetric to the first
# The frequency of bin b(i) is i * k where k = sample_rate / sample_size
sample_size = 2048

def get_bins(sample_rate, sample_size):
    idxs = np.array(range(0, int(sample_size / 2)))
    k = sample_rate / sample_size
    return idxs * k


# How many frames to step ahead after each sample
step_size = 1024

data_dir = 'training_data'
label_regex = "(?<=\\[)[a-z]{3}(?=\\])"

labels = []


for label in os.listdir(data_dir):
    # Filter out the readme and other non labels in dataset
    if (os.path.isdir(os.path.join(data_dir, label))):
        labels.append(label)

# Iterate over subfolders (labels)
for label in labels:
    # Iterate over wavfiles
    for filename in os.listdir(os.path.join(data_dir, label)):
        file_path = os.path.join(data_dir, label, filename)
        sample_rate, audio = wavfile.read(file_path)

        # Find labels by matching 3 letters between square brackets
        cur_labels = re.findall(label_regex, filename)
        
        # Combine channels and as stereo does not contain important information
        audio = audio[:, 0] + audio[:, 1]

        # Iterate through samples
        idx = 0
        while idx + sample_size < len(audio):

            # Get samples for this iteration
            samples = audio[idx : idx + sample_size]
            
            # Get first half of bins
            freqs = fft(samples)[0 : int(sample_size / 2)]

            # Result of fft is complex, np.abs returns magnitude
            freqs = np.abs(freqs)


            plt.plot(get_bins(sample_rate, sample_size), freqs)
            plt.xlim([0, 8192])
            plt.show()

            idx = idx + sample_size

        print(samples)



    
