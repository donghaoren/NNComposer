import os
import numpy as np
from audio_signals.audio_file import readAudioFile

supportedFormats = [
    ".mp3", ".m4a"
]

def generateDataset(output_path, tr, directories):
    filenames = set()
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                path = os.path.join(root, file)
                ext = os.path.splitext(path)[1].lower()

                if ext in supportedFormats:
                    data = readAudioFile(path, tr.rate)
                    spec = tr.transform(data)

                    fn_short = os.path.splitext(os.path.basename(path))[0].lower().replace(" ", "").replace(",", "").replace(".", "").replace("[", "").replace("]", "")
                    fn_short = fn_short[:20] + "_" + fn_short[-5:]
                    fn_short_c = fn_short
                    suffix = 1
                    while fn_short_c in filenames:
                        fn_short_c = fn_short + str(suffix)
                        suffix += 1

                    print fn_short_c + ".npy"
                    np.save(os.path.join(output_path, fn_short_c + ".npy"), spec)
                    filenames.add(fn_short_c)