import os
import re

import numpy as np
import tensorflow as tf

from scipy.io import wavfile
from scipy.fft import fft

from keras.models import Sequential
from keras.layers import Input, Dense


# WAV file values are 16-bit ranging from -32,768 to 32,767
# Sampling rate usually 44,100Hz
# So the max frequency bin of the FFT is sample_rate / 2 = 22,050
# The number of (useful) bins, n, is dependant on sample_size input into FFT divided by 2,
# the other half are symmetric to the first
# The frequency of bin b(i) is i * k where k = sample_rate / sample_size

sample_size = 1024

# Converts the bin number into the frequency it represents
def get_bins(sample_rate, sample_size):
    idxs = np.array(range(0, int(sample_size / 2)))
    k = sample_rate / sample_size
    return idxs * k

# Takes list of labels i.e. cla, cel and returns array of the same length as all_labels
# Returned array has values of 0 or 1 [0, 1, 0, 0, 0, 1, 0, ...]
def to_output_vector(labels: list, all_labels: list):
    result = np.zeros(len(all_labels), dtype=int)
    for label in labels:
        result[all_labels.index(label)] = 1
    return result


# How many frames to step ahead after each sample
step_size = int(sample_size / 2)

data_dir = 'training_data'
label_regex = "(?<=\\[)[a-z]{3}(?=\\])"

labels = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

# This would be useful if we were adding more instrument labels
# for label in os.listdir(data_dir):
#     # Filter out the readme and other non labels in dataset
#     if (os.path.isdir(os.path.join(data_dir, label))):
#         labels.append(label)


x_train = []
y_train = []

# Iterate over subfolders (labels) to gather training data
for label in labels:

    print("Processsing files in: " + label)
    # Iterate over wavfiles
    files = os.listdir(os.path.join(data_dir, label))
    for file_index, filename in enumerate(files):
        print("\rFile " + str(file_index + 1) + "/" + str(len(files)), end="")
        file_path = os.path.join(data_dir, label, filename)
        sample_rate, audio = wavfile.read(file_path)

        # Find labels by matching 3 letters between square brackets
        cur_labels = re.findall(label_regex, filename)

        # Some files have labels about genre which we can ignore
        cur_labels = list(filter(lambda l: l in labels, cur_labels))

        # Convert array of 1s and 0s
        cur_y = to_output_vector(cur_labels, labels)
        
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

            # Normalize between 0 and 1, skip if denominator = 0
            if np.max(freqs) - np.min(freqs) == 0:
                idx = idx + sample_size
                continue
            freqs_norm = (freqs - np.min(freqs)) / (np.max(freqs) - np.min(freqs))

            # Get training data
            x_train.append(freqs_norm)
            y_train.append(cur_y)
            idx = idx + sample_size
    print("\n")


# Create the model
model = Sequential()
input_size = int(sample_size / 2)
model.add(Input(shape=input_size))

# Add constantly decreasing in size layers, this should not affect prediction performance
# but should help computational performance
model.add(Dense(input_size, activation=tf.nn.relu))
model.add(Dense(int(input_size / 2), activation=tf.nn.relu))
model.add(Dense(int(input_size / 4), activation=tf.nn.relu))

model.add(Dense(len(labels), activation=tf.nn.sigmoid))

# Convert data to correct form
x_train = np.array(x_train)
y_train = np.array(y_train)

# Compile model and fit data
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='categorical_accuracy')
model.fit(x_train, y_train, validation_split=0.3, epochs=1, batch_size=32)

model.save("saved_model")

# All code below is just demoing the prediction on the first file
# Will make a method for the redundant input processing
test_file = os.path.join(data_dir, "cel", "[cel][cla]0001__1.wav")
test_sample_rate, test_audio = wavfile.read(test_file)
test_audio = test_audio[:, 0] + test_audio[:, 1]

idx = 0
while idx + sample_size < len(test_audio):
    test_sample = test_audio[idx : idx + sample_size]
    test_freqs = fft(test_sample)[0 : int(sample_size / 2)]
    test_freqs = np.abs(test_freqs)
    test_freqs_norm = (test_freqs - np.min(test_freqs)) / (np.max(test_freqs) - np.min(test_freqs))
    x_test = np.expand_dims(test_freqs_norm, axis=0)
    y_test = model.predict(x_test)
    idx = idx + step_size