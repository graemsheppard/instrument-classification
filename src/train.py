import os
import re

import numpy as np
import tensorflow as tf

from scipy.io import wavfile
from scipy.fft import fft

from keras.models import Sequential
from keras.layers import Input, Dense

from util import LABELS, SAMPLE_SIZE, STEP_SIZE, transform, to_output_vector


# WAV file values are 16-bit ranging from -32,768 to 32,767
# Sampling rate usually 44,100Hz
# So the max frequency bin of the FFT is sample_rate / 2 = 22,050
# The number of (useful) bins, n, is dependant on sample_size input into FFT divided by 2,
# the other half are symmetric to the first
# The frequency of bin b(i) is i * k where k = sample_rate / sample_size


data_dir = 'training_data'
label_regex = "(?<=\\[)[a-z]{3}(?=\\])"


x_train = []
y_train = []

# Iterate over subfolders (labels) to gather training data
for label in LABELS:

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
        cur_labels = list(filter(lambda l: l in LABELS, cur_labels))

        # Convert labels array of 1s and 0s
        cur_y = to_output_vector(cur_labels)

        # Iterate through samples
        idx = 0
        while idx + SAMPLE_SIZE < len(audio):

            # Get samples for this iteration
            samples = audio[idx : idx + SAMPLE_SIZE]
            
            # Apply fft and normalization
            freqs = transform(samples)
            if freqs is None:
                idx = idx + STEP_SIZE
                continue

            # Get training data
            x_train.append(freqs)
            y_train.append(cur_y)
            idx = idx + STEP_SIZE
    print("\n")


# Create the model
model = Sequential()
input_size = int(SAMPLE_SIZE / 2)
model.add(Input(shape=input_size))

# Add constantly decreasing in size layers, this should not affect prediction performance
# but should help computational performance
model.add(Dense(input_size, activation=tf.nn.relu))
model.add(Dense(int(input_size / 2), activation=tf.nn.relu))


model.add(Dense(len(LABELS), activation=tf.nn.sigmoid))

# Convert data to correct form
x_train = np.array(x_train)
y_train = np.array(y_train)

# Compile model and fit data
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='binary_accuracy')
model.fit(x_train, y_train, validation_split=0.3, epochs=3, batch_size=128)

model.save("saved_model")