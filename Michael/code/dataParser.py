
# This file will help handle data preprocessing
import scipy.io.wavfile as wavefile
import matplotlib.pyplot as plt
import numpy as np

import pywt

def wavLoader(filename='example.wav'):
    rate, data = wavefile.read(filename) 
    return rate, data

def wav2wpc(data):
    wavelet = 'db2'
    level = 4
    order = "freq"  # other option is "normal"
    interpolation = 'nearest'
    cmap = plt.cm.cool
    # Construct wavelet packet
    wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
    nodes = wp.get_level(level, order=order)
    labels = [n.path for n in nodes]
    waveletCoeffs = np.array([n.data for n in nodes], 'd')
    waveletCoeffs = abs(waveletCoeffs)
    # shannonEntropy = getShannonEntropy(waveletCoeffs)
    energyIndex = getEnergyIndices(waveletCoeffs)
    return energyIndex

def getEnergyIndices(waveletCoeffs):
    (coeffN, _) = waveletCoeffs.shape
    energyIndices = []
    # energyIndices = np.asarray(energyIndices)
    energyIndices = np.zeros((coeffN,1))
    for i, row in enumerate(waveletCoeffs):
        # aaa = np.dot(row, row)/len(row)
        energyIndices[i] = np.dot(row, row)/len(row)
    return energyIndices
