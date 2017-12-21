
# Michael


#######-----Import modules-----#######
from dataParser import wavLoader, wav2wpc, plot_signal_decomp, wav2dwtc
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import math
from tqdm import tqdm


#######-----Import, fragment, compute wavelet energy index-----#######
windowTS = 0.025
fs = 16000
windowLength = int(windowTS*fs)

filenames = glob.glob('275_candidates/*.wav')
counter = 0
speak_ct = 0

f = open('filenames', 'w')
f.write('\n'.join(map(lambda x: str(x), filenames)))
f.close()
badfilenames = []
testDataset = []
testLabels = []
trainDataset = []
trainLabels = []

rate, Data = wavLoader(filename='275_candidates/Marilu_Henner_test.wav')
nWindows = math.floor(len(Data)/windowLength)
nWindows = nWindows - nWindows%2000
print(nWindows)


for speak_ct, filename in enumerate(filenames):
    print(filename)
    try:
        rate, Data = wavLoader(filename=filename)
        nWindows = math.floor(len(Data)/windowLength)
        nWindows = nWindows - nWindows%2000
        print(nWindows)
        seqList = []
        for i in tqdm(range(int(nWindows))):
            data = Data[i*windowLength:(i+1)*windowLength]
            data = data.astype(np.int32)
            a_EI = wav2dwtc(data, w='db20', level=8)
            seqList.append(a_EI)
            if (i+1)%200 is 0:
                if 'test' in filename:
                    testDataset.append(seqList)
                    testLabels.append(speak_ct)
                elif 'train' in filename:
                    trainDataset.append(seqList)
                    trainLabels.append(speak_ct)
                seqList = []
                
    except:
        badfilenames.append(filename)


np.save('testDataset', np.squeeze(np.asarray(testDataset)))
np.save('testLabels', np.squeeze(np.asarray(testLabels)))
np.save('trainDataset', np.squeeze(np.asarray(trainDataset)))
np.save('trainLabels', np.squeeze(np.asarray(trainLabels)))

f = open('bad_filenames', 'w')
f.write('\n'.join(map(lambda x: str(x), badfilenames)))
f.close()
