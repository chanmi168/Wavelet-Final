
# This file will help handle data preprocessing
# from mutagen.mp3 import MP3
import scipy.io.wavfile as wavefile


def wavLoader(filename='example.wav'):
    rate, data = wavefile.read(filename)
    
    return rate, data