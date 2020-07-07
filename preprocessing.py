# import the libraries and modules
import os
from glob import glob
import pandas as pd
import numpy as np
import csv
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
#%matplotlib inline

# ***************************** let's examine a particular sound file (optional part) *****************************
# loading the file name of a one audio sample
file_name = os.path.abspath('UrbanSound8K_fold1/21684-9-0-7.wav')

# load the audio file
signal, sr = librosa.load(file_name)
print('shape is {}, sampling_rate is {}'.format(signal.shape, sr))
print('data type is {}', signal.dtype)

# playing the original audio file
ipd.Audio(file_name)

# visualization (Plot the audio signal)
plt.figure(figsize=(12, 4))
librosa.display.waveplot(signal, sr=sr)
# *****************************************************************************************

data_dir = 'UrbanSound8K_fold1/' # dataset directory
mfcc_ = [] # MFCCs array
class_ = [] # Classes array
fold_ = [] # folds array

# reading metadata file and extract audio files information.
df = pd.read_csv("UrbanSound8K.csv")

# because using just one fold, I filter the metadata file to include just audio files from fold one
# if you have all dataset folds, this part could be eliminated
filter_ = df['fold'] == 1
data = df[filter_]

# extract feature (MFCC) from a audio files
for i in range(data.shape[0]):
    slice_file_name = data.iat[i, 0]
    fold = data.iat[i, 5]
    classID = data.iat[i, 6]
    x, sample_rate = librosa.load(data_dir + slice_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
    mfcc_.append(mfccs)
    class_.append(classID)
    fold_.append(fold)

# writing the MFCC output to CSV file
# output is a CSV file which each row is MFCCs values for a single audio file
with open('mfccs.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(mfcc_)

# writing the class labels to CSV file
# output is a CSV file which each row is urban sound class for a single audio file
with open('labels.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows([[a] for a in class_])

# writing the folds number to CSV file
# output is a CSV file which each row is fold number for a single audio file
with open('folds.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows([[a] for a in fold_])