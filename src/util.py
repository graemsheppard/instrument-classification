import re

import numpy as np

from scipy.fft import fft
from scipy import interpolate

# Script to contain helper functions and variables

SAMPLE_RATE = 44100
SAMPLE_SIZE = 1024
STEP_SIZE = 512
LABELS = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

# Accepts a 2-channel audio segment of length SAMPLE_SIZE and performs fft and normalization
# Returns new array with transformed input
def transform(samples: np.ndarray):
    if len(samples) != SAMPLE_SIZE:
        raise RuntimeError("Length of provided array was " + str(len(samples)) + ", expected " + str(SAMPLE_SIZE))
    
    # Combine channels as stereo does not contain important information
    samples = samples[:, 0] + samples[:, 1]

    # Get first half of fft
    freqs = fft(samples)[0 : int(SAMPLE_SIZE / 2)]

    # Result of fft is complex, np.abs returns magnitude
    freqs = np.abs(freqs)

    # Normalize between 0 and 1, skip if denominator = 0
    if np.max(freqs) == 0:
        return None
    freqs_norm = freqs / np.max(freqs)

    return freqs_norm


# Converts the bin number into the frequency it represents
def get_bins():
    idxs = np.array(range(0, int(SAMPLE_SIZE / 2)))
    k = SAMPLE_RATE / SAMPLE_SIZE
    return idxs * k

# Takes list of labels i.e. cla, cel and returns array of the same length as LABELS which is all possible labels
# Returned array has values of 0 or 1 [0, 1, 0, 0, 0, 1, 0, ...]
def to_output_vector(labels: list):
    result = np.zeros(len(LABELS), dtype=int)
    for label in labels:
        result[LABELS.index(label)] = 1
    return result

def parse_labels(filename: str):
    label_regex = "(?<=\\[)[a-z]{3}(?=\\])"
    # Find labels by matching 3 letters between square brackets
    labels = re.findall(label_regex, filename)
    # Some files have labels about genre which we can ignore
    return list(filter(lambda l: l in LABELS, labels))

# Perform flat moving average smoothing
def smooth(arr: list, window: int):
    if (window % 2 == 0 or window <= 3):
        raise ValueError("Window must be an odd integer and greater than 3")
    
    result = []
    for i in range(len(arr)):
        sum = 0
        size = 0
        for j in range(window):
            idx = i - int(window / 2) + j
            if not (idx < 0 or idx >= len(arr)):
                sum += arr[idx]
                size += 1                

        result.append(sum / size)

    return np.array(result)


def get_exp_bins():
    bins = list(range(0, 512))
    result = []
    for b in bins:
        result.append(55 * pow(2, b * 0.01689))
    return result

def to_exponential(x):
    bins = get_bins()
    exp_bins = get_exp_bins()
    interpolater = interpolate.interp1d(bins, x)
    return interpolater(exp_bins)
    