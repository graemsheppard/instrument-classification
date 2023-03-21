import numpy as np

from scipy.fft import fft

# Script to contain helper functions and variables

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
    if np.max(freqs) - np.min(freqs) == 0:
        return None
    freqs_norm = (freqs - np.min(freqs)) / (np.max(freqs) - np.min(freqs))

    return freqs_norm


# Converts the bin number into the frequency it represents
def get_bins(sample_rate, sample_size):
    idxs = np.array(range(0, int(sample_size / 2)))
    k = sample_rate / sample_size
    return idxs * k

# Takes list of labels i.e. cla, cel and returns array of the same length as LABELS which is all possible labels
# Returned array has values of 0 or 1 [0, 1, 0, 0, 0, 1, 0, ...]
def to_output_vector(labels: list):
    result = np.zeros(len(LABELS), dtype=int)
    for label in labels:
        result[LABELS.index(label)] = 1
    return result