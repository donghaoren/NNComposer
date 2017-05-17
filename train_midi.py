import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file
from model import build_midi_model

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session():
    gpu_options = tf.GPUOptions(allow_growth = True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

length = 512

print('Loading data...')

data = np.load("midi_rolls.npy")

print('Vectorization...')
X = np.zeros((len(data) / 2, length), dtype=np.bool)
for i in range(0, len(data), 2):
    X[i / 2, data[i]] = 1
    X[i / 2, data[i + 1]] = 1

    
with tf.device("/gpu:1"):
    nbatch = 32
    maxlen = 16
    model = build_midi_model(nbatch, maxlen)
    
import random
def generateXYChunk(data, batch_size, chunk_length, time_steps, bins):
    X = np.zeros((chunk_length / time_steps * batch_size, time_steps, bins))
    Y = np.zeros((chunk_length / time_steps * batch_size, time_steps, bins))
    for i in range(batch_size):
        while True:
            if data.shape[0] - chunk_length - 2 < 1: continue
            s = random.randint(0, data.shape[0] - chunk_length - 2)
            break
            
        for j in range(chunk_length / time_steps):
            X[i + j * batch_size,:,:] = data[s:s+time_steps,:]
            Y[i + j * batch_size,:,:] = data[s+1:s+1+time_steps,:]
            s += time_steps
    return X, Y

from keras.callbacks import ModelCheckpoint
XX_validation, YY_validation = generateXYChunk(X, nbatch, 4096, maxlen, length)
checkpointer = ModelCheckpoint(filepath="midi-weights.hdf5", verbose=1, save_best_only=True)

current_loss = 1e10

print('Training...')

# model.load_weights("text.h5")
for iii in range(0, 100000):
    XXs, YYs = generateXYChunk(X, nbatch, 65536, maxlen, length)
    print(XXs.shape, YYs.shape)
    model.reset_states()
    model.fit(XXs, YYs, batch_size=nbatch, epochs=iii+1, shuffle=False, verbose = 1, initial_epoch = iii)
    
    model.reset_states()
    loss = model.evaluate(XX_validation, YY_validation, batch_size=nbatch)
    
    print "\n#### Epoch: %04d - Validation loss = %04.2f" % (iii, loss)
    
    if loss < current_loss:
        model.save("midi_models/midi-%06d-%04.2f.h5" % (iii, loss))
        current_loss = loss
