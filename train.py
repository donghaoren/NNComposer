from datasets.generate import generateDataset
from audio_signals.stft import STFTTransform
import numpy as np
import os
import random
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session():
    gpu_options = tf.GPUOptions(allow_growth = True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

from model import build_lstm_model_512

batch_size = 8
time_steps = 64
model = build_lstm_model_512(batch_size, time_steps)

def getDataFiles(directory = "data"):
    result = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            path = os.path.join(root, f)
            if os.path.splitext(path)[1].lower() == ".npy":
                result.append(path)
    return result

files = getDataFiles()

for epoch in range(10000):
    for filename in files:
        print "# Epoch %d - %s" % (epoch, filename)
        data = np.load(filename)
        data = np.log(1e-6 + data)
        
        bins = data.shape[1]
        
        # Prepare the data in to (batch_size * K, time_steps, freq_bins) shape
        K = 128
        X = np.zeros((batch_size * K, time_steps, bins))
        Y = np.zeros((batch_size * K, bins))
        for i in range(batch_size):
            s = random.randint(0, data.shape[0] / K / time_steps - 1)
            for j in range(K):
                X[i + j * batch_size,:,:] = data[s:s+time_steps,:]
                Y[i + j * batch_size,:] = data[s+time_steps,:]
                s += time_steps

        print X.shape, Y.shape, model.input_shape, model.output_shape
                
        model.reset_states()
        model.fit(X, Y, epochs=1, shuffle=False, batch_size=batch_size)
    
        model.save("models/epoch-%06d.hdf5" % epoch)