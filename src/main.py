import os
import train
import argparse
from isolate import isolate

import numpy as np
import tensorflow as tf

from scipy.fft import fft
from scipy.io import wavfile
from keras.models import load_model
from keras.engine.sequential import Sequential

from util import SAMPLE_SIZE, STEP_SIZE, LABELS, transform

def parse_args():
    parser = argparse.ArgumentParser(description="An AI that detects the instruments playing in a sample")
    parser.add_argument('-p', '--path', type=str, help='Input file path')
    parser.add_argument('-b', '--build', action='store_true', help='Force build the model, even if one already exists')
    parser.add_argument('-a', '--all', action='store_true', help='Predict for all audio files in the set.')
    args = parser.parse_args()
    return args
    
def load_keras_model():
    model: Sequential = load_model('saved_model')
    return model
    
def predict_files(model):
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
         
        averages = predict(audio_file, model)
        
        print(f'Labels: {labels}')
        print('Predicted:')
        print(averages)

        for label in get_labels(averages):
            print(label)
                
def predict(path, model):
    test_sample_rate, test_audio = wavfile.read(path)

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

        sums = [sums[i] + value for i, value in enumerate(y_test[0])]
        idx = idx + STEP_SIZE
        
    averages = [sum/num_samples for sum in sums]
       
    return(averages)
    
def get_labels(averages):
    labels = []
    for i, average in enumerate(averages):
        if average > 0.2:
            labels.append(LABELS[i])
    return labels

def main():
    args = parse_args()
    
    # Build the model if requested
    if (args.build):
        train.main()
    
    # If the model does not already exist, build it first
    try:
        model = load_keras_model()
    except:
        train.main()
        model = load_keras_model()

    # If we want all the files eg for a sample 
    if (args.all):
        predict_files(model)
        return
        
    # Otherwise, we predict from the specified file. Here's a sample file from the data set that can quickly be used as a default.
    input_file_path = os.path.join('training_data', 'tru', '[tru][cla]1870__1.wav')
    if not (args.path is None or args.path == ''):
        input_file_path = args.path
    
    labels = get_labels(predict(input_file_path, model))
    
    print('Instruments found:')
    for i in range(len(labels)):
        print(f' {i+1}. {labels[i]}')
        
    instrument = input('Please enter the number of the instrument you\'d like to isolate:\n>> ')
    instrument = int(instrument) - 1
    
    outfile = input('Please enter the name of the file to output (default: output.wav):\n>> ')
    if (outfile is None or outfile == ''):
        outfile = 'output.wav'
    
    isolate(labels[instrument], input_file_path, output_path=outfile)
    
if __name__ == "__main__":
    main()
    