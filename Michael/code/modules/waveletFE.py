
# Michael


#######-----Import modules-----#######
from dataParser import wavLoader, wav2wpc, plot_signal_decomp, wav2dwtc, getDataNLabels
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import math





#######-----Import, fragment, compute wavelet energy index-----#######

foldername = '/Users/MichaelChan/Desktop/JHU/Wavelets and Filter Banks/Final Project/275_candidates'
filenames = glob.glob(foldername+'/*.wav')
# filenames = glob.glob('275_candidates/*.wav')
filenames = sorted(filenames, key =lambda x: x.split('/')[-1])

f = open('filenames', 'w')
f.write('\n'.join(map(lambda x: str(x), filenames)))
f.close()


testDataset, testLabels, trainDataset, trainLabels, badfilenames = getDataNLabels(filenames)

np.save('275_candidates/testDataset', np.squeeze(np.asarray(testDataset)))
np.save('275_candidates/testLabels', np.squeeze(np.asarray(testLabels)))
np.save('275_candidates/trainDataset', np.squeeze(np.asarray(trainDataset)))
np.save('275_candidates/trainLabels', np.squeeze(np.asarray(trainLabels)))

f = open('bad_filenames', 'w')
f.write('\n'.join(map(lambda x: str(x), badfilenames)))
f.close()
