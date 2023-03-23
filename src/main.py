import os

import numpy as np
import tensorflow as tf

from scipy.fft import fft
from scipy.io import wavfile
from keras.models import load_model
from keras.engine.sequential import Sequential

from util import SAMPLE_SIZE, STEP_SIZE, LABELS, transform, parse_labels, to_output_vector


def main():

    # All code below is just demoing the prediction on the training data

    data_dir = "training_data"

    step_size = 16 * STEP_SIZE

    model: Sequential = load_model('saved_model')

    # Print 3 decimals
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    passes = 0
    for label in reversed(LABELS):
        files = os.listdir(os.path.join(data_dir, label))
        for file_index, filename in enumerate(files):
            file_path = os.path.join(data_dir, label, filename)
            sample_rate, audio = wavfile.read(file_path)
            idx = 0
            y_test = to_output_vector(parse_labels(filename))
            last_n = []
            while idx + SAMPLE_SIZE < len(audio):
                samples = audio[idx : idx + SAMPLE_SIZE, :]
                freqs = transform(samples)
                x_test = np.expand_dims(freqs, axis=0)
                y_pred = model.predict(x_test, verbose=0)[0]
                
                last_n.append(y_pred)
                if (len(last_n) > 64):
                    last_n.pop(0)
                
                average = np.average(np.array(last_n), axis=0)

                y_pred = np.array(np.round(average, 0), dtype=int)
                if np.array_equal(y_test, y_pred):
                    print("PASS " + str(passes))
                    print(average)
                    passes = passes + 1

                idx = idx + step_size
        

if __name__ == "__main__":
    main()
    