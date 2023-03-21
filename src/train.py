import os
import re

from scipy.io import wavfile


# WAV file values are 16-bit ranging from -32,768 to 32,767

data_dir = 'training_data'
regex = "(?<=\\[)[a-z]{3}(?=\\])"

labels = []


for label in os.listdir(data_dir):
    # Filter out the readme and other non labels in dataset
    if (os.path.isdir):
        labels.append(label)

# Iterate over subfolders (labels)
for label in labels:
    # Iterate over wavfiles
    for filename in os.listdir(os.path.join(data_dir, label)):
        file_path = os.path.join(data_dir, label, filename)
        sample_rate, samples = wavfile.read(file_path)

        # Find labels by matching 3 letters between square brackets
        cur_labels = re.findall(regex, filename)
        
        # Combine and average channels as stereo does not contain important information
        samples = (samples[:, 0] + samples[:, 1]) / 2
        print(samples)


    
