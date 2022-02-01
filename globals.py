PATH_TO_MEOW = './data/meow/'
MIN_DURATION = 1.5
STFT_SHAPE = (1025, 65)


class AudioTooShortException(Exception):
    """Raised when audio file duration is not long enough."""
    pass
