import os

import numpy as np
import tensorflow as tf

from scipy.fft import fft
from scipy.io import wavfile
from keras.models import load_model
from keras.engine.sequential import Sequential

from util import SAMPLE_SIZE, STEP_SIZE, transform


def main():

    # All code below is just demoing the prediction on the first file

    data_dir = "training_data"

    test_file = os.path.join(data_dir, "cel", "[cel][cla]0001__1.wav")
    test_sample_rate, test_audio = wavfile.read(test_file)

    model: Sequential = load_model('saved_model')

    idx = 0
    while idx + SAMPLE_SIZE < len(test_audio):
        test_samples = test_audio[idx : idx + SAMPLE_SIZE, :]
        test_freqs = transform(test_samples)
        x_test = np.expand_dims(test_freqs, axis=0)
        y_test = model.predict(x_test, verbose=0)
        idx = idx + STEP_SIZE

if __name__ == "__main__":
    main()
    