import librosa
import os
import numpy as np
from tqdm import tqdm

from globals import PATH_TO_MEOW, MIN_DURATION, STFT_SHAPE, AudioTooShortException


def load_audio(fname, dur=MIN_DURATION):
    """Load and clip audio to be exactly MIN_DURATION long (or ignore if this is not possible)."""
    data, sr = librosa.load(os.path.join(PATH_TO_MEOW, fname), duration=dur)
    if librosa.get_duration(data, sr) < MIN_DURATION:
        raise AudioTooShortException
    return data, sr


def compute_stft(data):
    return librosa.stft(data)


def extract_features(data, sr):
    features = {}

    mean_amplitude = np.mean(np.array(data))
    features['Mean Amplitude'] = mean_amplitude

    rmse = librosa.feature.rms(data)
    mean_rmse = np.mean(rmse)
    features['Root Mean Square Energy'] = mean_rmse

    zero_crossings = librosa.zero_crossings(data, pad=False)
    zcr = sum(zero_crossings)
    features['Zero Crossing Rate'] = zcr

    spectral_centroids = librosa.feature.spectral_centroid(data, sr)
    mean_spectral_centroid = np.mean(spectral_centroids)
    features['Mean Spectral Centroid'] = mean_spectral_centroid

    spectral_rolloff = librosa.feature.spectral_rolloff(data, sr)
    mean_spectral_rolloff = np.mean(spectral_rolloff)
    features['Mean Spectral Rolloff'] = mean_spectral_rolloff

    mfccs = librosa.feature.mfcc(data, sr)
    mean_mfccs = np.mean(mfccs, axis=1)
    for idx, mean_mfcc in enumerate(mean_mfccs):
        features[f'Mean MFCC {idx}'] = mean_mfcc

    return features


def preprocess_audio():
    fnames = os.listdir(PATH_TO_MEOW)

    audios = []

    for fname in tqdm(fnames):
        try:
            data, _ = load_audio(fname)
            stft = compute_stft(data)
            stft = np.reshape(stft, (1, stft.shape[0], stft.shape[1]))
            audios.append(stft)
        except AudioTooShortException:
            continue

    return audios


if __name__ == '__main__':
    preprocess_audio()
