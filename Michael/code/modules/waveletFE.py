
# Michael


#######-----Import modules-----#######
from dataParser import wavLoader, wav2wpc, plot_signal_decomp, wav2dwtc, getDataNLabels
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import math
import argparse

#######-----Parse data-----#######
parser = argparse.ArgumentParser(description='Process some audio signal.')
parser.add_argument('--WAVELET_DIM', type=int, default=8, metavar='LR',
                    help='WAVELET_DIM (default: 8)')
parser.add_argument('--SEQUENCE_DIM', type=int, default=200, metavar='LR',
                    help='SEQUENCE_DIM (default: 200)')
parser.add_argument('--SEQUENCE_TIME', type=float, default=5, metavar='LR', 
                    help='SEQUENCE_TIME (default: 5)')
parser.add_argument('--BATCH_DIM', type=int, default=10, metavar='LR', 
                    help='BATCH_DIM (default: 10)')

args = parser.parse_args()
WAVELET_DIM=args.WAVELET_DIM
SEQUENCE_DIM=args.SEQUENCE_DIM
SEQUENCE_TIME=args.SEQUENCE_TIME
BATCH_DIM=args.BATCH_DIM


#######-----Import, fragment, compute wavelet energy index-----#######

foldername = '/Users/MichaelChan/Desktop/JHU/Wavelets and Filter Banks/Final Project/275_candidates'
filenames = glob.glob(foldername+'/*.wav')
# filenames = glob.glob('275_candidates/*.wav')
filenames = sorted(filenames, key =lambda x: x.split('/')[-1])

f = open('filenames', 'w')
f.write('\n'.join(map(lambda x: str(x), filenames)))
f.close()


testDataset, testLabels, trainDataset, trainLabels, badfilenames = getDataNLabels(filenames, WAVELET_DIM, SEQUENCE_DIM, SEQUENCE_TIME, BATCH_DIM)
# testDataset, testLabels, trainDataset, trainLabels, badfilenames = getDataNLabels(filenames)

np.save('275_candidates/testDataset', np.squeeze(np.asarray(testDataset)))
np.save('275_candidates/testLabels', np.squeeze(np.asarray(testLabels)))
np.save('275_candidates/trainDataset', np.squeeze(np.asarray(trainDataset)))
np.save('275_candidates/trainLabels', np.squeeze(np.asarray(trainLabels)))

f = open('bad_filenames', 'w')
f.write('\n'.join(map(lambda x: str(x), badfilenames)))
f.close()

print('WAVELET_DIM =', WAVELET_DIM)
print('SEQUENCE_DIM =', SEQUENCE_DIM)
print('SEQUENCE_TIME =', SEQUENCE_TIME)
print('BATCH_DIM =', BATCH_DIM)