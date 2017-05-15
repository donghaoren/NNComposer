from scipy.signal import stft, istft
import numpy as np
import math

class STFTTransform:
    def __init__(self, window_size = 2048, size = 512, rate = 44100):
        """ Initialize the STFTTransformer.

        Args:
            window_size: STFT size, default 2048
            size: Output feature size, default 512
            rate: Input audio sample rate, default 44100
        """

        self.rate = rate
        self.window_size = window_size
        self.size = size
        self.noverlap = self.window_size - self.window_size / 16
        self.window_advance = self.window_size - self.noverlap

    def transform(self, audio):
        """ Transform audio into STFT features.

        Args:
            audio: The audio signal

        Returns:
            numpy array of features with shape (slices, size)
        """

        f, t, Zxx = stft(audio, self.rate, window = "hann", nperseg = self.window_size, noverlap = self.noverlap, return_onesided = True)
        return np.abs(Zxx)[1:1+self.size,:].T

    def invert(self, data, iterations = 10):
        """ Transform STFT features back to audio.

        Args:
            data: STFT features with shape (slices, size)
            iterations: Number of iterations for phase correction

        Returns:
            numpy array of reconstructed signal
        """

        data = data.T
        data = np.vstack([
            np.zeros((1, data.shape[1])),
            data,
            np.zeros((self.window_size / 2 + 1 - self.size - 1, data.shape[1]))
        ])
        length = data.shape[1] * (self.window_size - self.noverlap) - (self.window_size - self.noverlap)
        x = np.random.normal(0, 1, (length,))
        for i in range(iterations):
            f, t, Zxx = stft(x, self.rate, window = "hann", nperseg = self.window_size, noverlap = self.noverlap, return_onesided = True)
            Z = np.exp(np.complex(0, 1) * np.angle(Zxx)) * data
            t, x = istft(Z, self.rate, window = "hann", nperseg = self.window_size, noverlap = self.noverlap, input_onesided = True)
        return np.real(x)
