import os

import numpy as np
import tensorflow as tf

from scipy.fft import fft
from scipy.io import wavfile
from keras.models import load_model
from keras.engine.sequential import Sequential

from util import SAMPLE_SIZE, STEP_SIZE, LABELS, transform


def main():
    model: Sequential = load_model('saved_model')

    data_dir = "testing_data"
    files = os.listdir(data_dir)
    
    print(files)

    for idx, file in enumerate(files):
        extension = file.split('.')[1]
        if extension == 'wav':
            continue

        print('--------------------------------------------------')

        labels = []
        with open(os.path.join(data_dir,file), "r") as label_file:
            data = label_file.read()
            lines = data.split('\n')
            labels = [label for label in lines]
        
        audio_file = os.path.join(data_dir, f'{file.split(".")[0]}.wav')
         
        test_sample_rate, test_audio = wavfile.read(audio_file)

        sums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        idx = 0
        num_samples = 0
        while idx + SAMPLE_SIZE < len(test_audio):
            num_samples+=1
            test_samples = test_audio[idx : idx + SAMPLE_SIZE, :]
            test_freqs = transform(test_samples)
            x_test = np.expand_dims(test_freqs, axis=0)
            try:
                y_test = model.predict(x_test, verbose=0)
            except Exception as e:
                print(e)
                break

            sums = [averages[i] + value for i, value in enumerate(y_test[0])]
            idx = idx + STEP_SIZE
        
        averages = [sum/num_samples for sum in sums]
        
        print(f'Labels: {labels}')
        print('Predicted:')
        print(averages)

        for i, average in enumerate(averages):
            if average > 0.2:
                print(LABELS[i])

if __name__ == "__main__":
    main()
    