
# This file will help handle data preprocessing
import scipy.io.wavfile as wavefile
import matplotlib.pyplot as plt
import numpy as np

import pywt
import pywt.data
import sys
from tqdm import tqdm
import math

def wavLoader(filename='example.wav'):
    rate, data = wavefile.read(filename) 
    return rate, data

def wav2wpc(data):
    wavelet = 'db20'
    level = 4
    order = "freq"  # other option is "normal"
    interpolation = 'nearest'
    cmap = plt.cm.cool
    # Calculate wavelet packet
    wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
    nodes = wp.get_level(level, order=order)
    labels = [n.path for n in nodes]
    waveletCoeffs = np.array([n.data for n in nodes], 'd')
    waveletCoeffs = abs(waveletCoeffs)
    # shannonEntropy = getShannonEntropy(waveletCoeffs)
    energyIndex = getEnergyIndices(waveletCoeffs)
    return energyIndex

def getEnergyIndices(waveletCoeffs):
    coeffN = len(waveletCoeffs)
    energyIndices = np.zeros((coeffN,1))
    for i, row in enumerate(waveletCoeffs):
        energyIndices[i] = np.dot(row, row)/len(row)
    return energyIndices


def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.

    S = An + Dn + Dn-1 + ... + D1
    """
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(20):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    a_EI = getEnergyIndices(rec_a)
    d_EI = getEnergyIndices(rec_d)

    fig = plt.figure()
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))

    return  a_EI, d_EI

def wav2dwtc(data, w, level):
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)
    # a = data
    # normalizee data before processsings
    eps = sys.float_info.epsilon
    a = (data-min(data)+eps)/(max(data)-min(data)+eps)
    ca = []
    cd = []

    rec_a = []

    for i in range(level):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)

    a_EI = getEnergyIndices(ca)
    d_EI = getEnergyIndices(cd)
    energyIdx = np.concatenate((a_EI, d_EI), axis=0)
    return  energyIdx


def getDataNLabels(filenames,WAVELET_DIM=8, SEQUENCE_DIM=200, SEQUENCE_TIME=5, BATCH_DIM=10):

    # print('WAVELET_DIM =', WAVELET_DIM)
    # print('SEQUENCE_DIM =', SEQUENCE_DIM)
    # print('SEQUENCE_TIME =', SEQUENCE_TIME)
    # print('BATCH_DIM =', BATCH_DIM)


    windowTS = SEQUENCE_TIME/SEQUENCE_DIM
    fs = 16000
    windowLength = int(windowTS*fs)
    # print('windowLength is:', windowLength)
    # print('windowTS is:', windowTS)
    idxLabels = np.array(range(int(len(filenames)/2)))
    idxLabels = np.repeat(idxLabels, 2)
    labelDict = dict(zip(filenames, idxLabels))
    
    badfilenames = []
    testDataset = []
    testLabels = []
    trainDataset = []
    trainLabels = []


    for speak_ct, filename in enumerate(filenames):
        speakerIdx = labelDict[filename]
        print(filename)
        try:
            rate, Data = wavLoader(filename=filename)
            nWindows = math.floor(len(Data)/windowLength)
            nWindows = nWindows - nWindows%int(SEQUENCE_DIM*BATCH_DIM)
            print(nWindows)

            seqList = []
            for i in tqdm(range(int(nWindows))):
                data = Data[i*windowLength:(i+1)*windowLength]
                data = data.astype(np.int32)
                energyIdx = wav2dwtc(data, w='db20', level=WAVELET_DIM)
                seqList.append(energyIdx)
                if (i+1)%SEQUENCE_DIM is 0:
                    if 'test' in filename:
                        testDataset.append(seqList)
                        testLabels.append(speakerIdx)
                    elif 'train' in filename:
                        trainDataset.append(seqList)
                        trainLabels.append(speakerIdx)
                    seqList = []
        except:
            badfilenames.append(filename)

    if len(badfilenames) is 0:
        badfilenames.append('empty')

    return testDataset, testLabels, trainDataset, trainLabels, badfilenames