from datasets.generate import generateDataset
from audio_signals.stft import STFTTransform

tr = STFTTransform(2048, 512, 44100)

dataset = generateDataset("data", tr, [
    "./audio"
])