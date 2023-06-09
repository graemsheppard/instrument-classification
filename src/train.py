import os

import numpy as np
import tensorflow as tf

from scipy.io import wavfile
from scipy import signal

from keras.models import Sequential
from keras.layers import Input, Dense

from matplotlib import pyplot as plt

from util import *

def main():
    # WAV file values are 16-bit ranging from -32,768 to 32,767
    # Sampling rate usually 44,100Hz
    # So the max frequency bin of the FFT is sample_rate / 2 = 22,050
    # The number of (useful) bins, n, is dependant on sample_size input into FFT divided by 2,
    # the other half are symmetric to the first
    # The frequency of bin b(i) is i * k where k = sample_rate / sample_size

    # TODO: kmeans to determine frequencies we need to isolate per label?

    data_dir = 'training_data'


    # Reduce the size of the training data for faster optimizations
    development = True

    # Step through files faster in development mode, taking fewer samples (still of length SAMPLE_SIZE)
    step_size = 8 * STEP_SIZE if development else STEP_SIZE

    train = []

    instruments = {}
    for label in LABELS:
        instruments[label] = []


    # Iterate over subfolders (labels) to gather training data
    for label in LABELS:
        # The number of files processed, should equal total files when development = False
        file_count = 0
        # Iterate over wavfiles
        files = os.listdir(os.path.join(data_dir, label))
        for file_index, filename in enumerate(files):
            # Take every 4th file in development mode
            if development and file_index % 16 != 0:
                continue

            print("\rProcesssing files in: " + label + " (" + str(file_count + 1) + "/" + str(len(files)) + ")...", end="")
            file_path = os.path.join(data_dir, label, filename)
            sample_rate, audio = wavfile.read(file_path)

            # Skip file if incorrect sample rate
            if sample_rate != SAMPLE_RATE:
                continue

            # Convert labels array of 1s and 0s
            cur_labels = parse_labels(filename)
            cur_y = to_output_vector(cur_labels)

            # Iterate through samples
            idx = 0
            while idx + SAMPLE_SIZE < len(audio):

                # Get samples for this iteration
                samples = audio[idx : idx + SAMPLE_SIZE]
                
                # Apply fft and normalization
                freqs = transform(samples)
                if freqs is None:
                    idx = idx + step_size
                    continue

                # Get training data
                row = []
                row.extend(cur_y)
                row.extend(freqs)
                train.append(row)

                for c in cur_labels:
                    instruments[c].append(freqs)

                idx += step_size
            file_count += 1

        print("DONE")

    avg_each_inst = []
    # Average all the samples per instrument
    for key, value in instruments.items():
        avg = np.sum(value, axis=0) / len(value)
        instruments[key] = avg
        avg_each_inst.append(avg)

    # Find the average frequencies across all instruments
    avg_all_inst = np.sum(avg_each_inst, axis=0) / len(LABELS)

    bins = get_bins()
    exp_bins = get_exp_bins()

    inst_freqs = {}
    # Subtract the average over all instruments from the current instrument to find dominant frequencies
    with open("inst_freqs.csv", "w+") as csv:
        header = "INST"
        for bin in exp_bins:
            header += ", " + str(int(bin))
        csv.write(header + "\n")
        for key, value in instruments.items():
            dominant_freqs = value - avg_all_inst
            # Restrict values to positive range
            dominant_freqs = np.clip(dominant_freqs, 0, 1)
            dominant_freqs = to_exponential(dominant_freqs)
            # Apply butterworth filter to reduce noise
            nyq = SAMPLE_RATE / 4
            cutoff = 800 / nyq
            sos = signal.butter(5, cutoff, 'lowpass', output='sos')
            dominant_freqs = signal.sosfilt(sos, dominant_freqs)
            # Normalize and cutoff at 0.2
            dominant_freqs = dominant_freqs / np.max(dominant_freqs)
            dominant_freqs = np.where(dominant_freqs < 0.2, 0, dominant_freqs)
            csv_row = key
            for freq in dominant_freqs:
                csv_row += ", " + str(freq)
            csv.write(csv_row + "\n")
            plt.plot(exp_bins, dominant_freqs, label=key)
    plt.legend()
    plt.show()

    # Create the model
    model = Sequential()
    input_size = int(SAMPLE_SIZE / 2)
    model.add(Input(shape=input_size))

    # Add constantly decreasing in size layers, this should not affect prediction performance
    # but should help computational performance
    model.add(Dense(int(input_size), activation=tf.nn.leaky_relu))
    model.add(Dense(int(input_size / 2), activation=tf.nn.leaky_relu))
    model.add(Dense(len(LABELS), activation=tf.nn.sigmoid))

    # Convert data to correct form and shuffle
    train = np.array(train)
    np.random.shuffle(train)
    x_train = train[:, 11:]
    y_train = train[:, :11]

    # Compile model and fit data
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.fit(x_train, y_train, validation_split=0.3, epochs=3, batch_size=128)

    model.save("saved_model")


if __name__ == "__main__":
    main()