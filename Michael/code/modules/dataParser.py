
# This file will help handle data preprocessing
import scipy.io.wavfile as wavefile
import matplotlib.pyplot as plt
import numpy as np

import pywt
import pywt.data

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
    a = (data-min(data))/(max(data)-min(data))
    ca = []
    cd = []

    rec_a = []

    for i in range(level):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        # coeff_list = [a, None] + [None] * i
        # rec_a.append(pywt.waverec(coeff_list, w))

    # rec_a_EI = getEnergyIndices(rec_a)
    a_EI = getEnergyIndices(ca)

    return  a_EI