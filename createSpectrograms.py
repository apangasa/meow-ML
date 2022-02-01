import librosa
import cv2
import os
import numpy as np

from globals import PATH_TO_MEOW


def generate_spectrogram(fname):
    data, sr = librosa.load(PATH_TO_MEOW + fname)
    spectrogram = librosa.feature.melspectrogram(data, sr)


if __name__ == '__main__':
    for fname in os.listdir(PATH_TO_MEOW):
        generate_spectrogram(fname)
