
# This file will help handle data preprocessing
from mutagen.mp3 import MP3
import scipy.io.wavfile as wavefile


def MP3loader (filename = 'example.mp3'):
    rate, data = wavefile.read(filename)

    return data