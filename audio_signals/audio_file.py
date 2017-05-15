import os
import subprocess
import numpy

def readAudioFile(filename, rate = 44100):
    proc = subprocess.Popen([
        "ffmpeg",
        "-i", filename,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(rate),
        "-f", "s16le",
        "-"
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    data = numpy.fromstring(proc.stdout.read(), numpy.int16)
    data = data.astype('float32') / 32767.0

    return data