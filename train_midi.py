# Allow gpu memory growth
from utils.gpu_config import tfSessionAllowGrowth
tfSessionAllowGrowth()

import time
import random
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
import keras.backend.tensorflow_backend as KTF

from midi_models.models import buildModel, getModelProperties
from midi_signals.midi import encodedMessageToVector
from utils.file_manager import FileManagerFS

fm = FileManagerFS("output")


# Which model to train
model_name = "Layer5_1024_256_1024"

# Device to train the model
device = "/gpu:0"

# Dataset
dataset = "data/midi_piano_rolls_dataset.pkl"

# Batch size (number of tracks in training/validation)
nbatch = 64

# Length of each run, also the length of the truncated back propagation
maxlen = 128

# Initial learning rate
learning_rate = 0.00000001

# Skip this number of samples before training / validation
preamble = 16 * nbatch # feed through the first # of samples before training/testing

# Number of timesteps per epoch
epoch_timesteps = 65536

# Validation size
validation_timesteps = 8192

# Reset model state every # batch
reset_states_every = epoch_timesteps / nbatch

load_checkpoint = None
load_checkpoint = "old/1495333001-midi-000174-0.555144.h5"


print "Build model..."

with tf.device(device):
    model = buildModel(model_name, nbatch, maxlen)
    model_props = getModelProperties(model_name)

enable_doubleinput = (model_props["input_format"] == "double_slice")


# Input is 180 dimensional vector
length = 180


print "Loading data..."

import pickle
with open(dataset, "rb") as f:
    data = pickle.load(f)


print "Vectorization..."

if enable_doubleinput:
    X = np.zeros((len(data) * 2, length), dtype=np.float32)
    for i in range(0, len(data)):
        X[i * 2,:] = encodedMessageToVector(data[i])
        X[i * 2 + 1,:] = encodedMessageToVector(data[i])
    print "#### Using double slice input format"
else:
    X = np.zeros((len(data), length), dtype=np.float32)
    for i in range(0, len(data)):
        X[i,:] = encodedMessageToVector(data[i])
    print "#### Using single slice input format"
    
def generateXYChunk(data, batch_size, chunk_length, time_steps, bins):
    X = np.zeros((chunk_length / time_steps * batch_size, time_steps, bins))
    if enable_doubleinput:
        Y = np.zeros((chunk_length / time_steps * batch_size, time_steps, bins + 1))
        mask = [ 0, 1 ] * (time_steps / 2)
        maskInv = [ 1, 0 ] * (time_steps / 2)
    else:
        Y = np.zeros((chunk_length / time_steps * batch_size, time_steps, bins))
    for i in range(batch_size):
        while True:
            if data.shape[0] - chunk_length - 2 < 1: continue
            s = random.randint(0, data.shape[0] - chunk_length - 3)
            break
            
        for j in range(chunk_length / time_steps):
            X[i + j * batch_size,:,:] = data[s:s+time_steps,:]
            Y[i + j * batch_size,:,0:180] = data[s+2:s+2+time_steps,:]
            if enable_doubleinput:
                Y[i + j * batch_size,:,180] = mask
                X[i + j * batch_size,:,0] *= maskInv
                X[i + j * batch_size,:,1] *= maskInv
                X[i + j * batch_size,:,2] *= maskInv
            s += time_steps
    return X, Y

class ResetStatesCallback(Callback):
    def __init__(self, max_len):
        self.counter = 0
        self.max_len = max_len

    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.max_len == 0:
            self.model.reset_states()
            print "state reset"
        self.counter += 1


XX_validation, YY_validation = generateXYChunk(X, nbatch, validation_timesteps, maxlen, length)

current_loss = 1e10
last_checkpoint = None


print('Training...')

session_timestamp = int(time.time())

if load_checkpoint is not None:
    fm.loadModel(model, load_checkpoint)
    last_checkpoint = load_checkpoint
    model.reset_states()
    model.evaluate(XX_validation[:preamble], YY_validation[:preamble], batch_size=nbatch, verbose=0)
    current_loss = model.evaluate(XX_validation[preamble:], YY_validation[preamble:], batch_size=nbatch, verbose=0)
    print "Loaded weights from %s: Learning rate = %.8f, Validation loss = %08.6f" % (load_checkpoint, learning_rate, current_loss)

KTF.set_value(model.optimizer.lr, learning_rate)

try:
    
    for iii in range(0, 100000):
        
        XXs, YYs = generateXYChunk(X, nbatch, epoch_timesteps, maxlen, length)

        model.reset_states()
        model.evaluate(XXs[:preamble], YYs[:preamble], batch_size=nbatch, verbose=0)
        model.fit(
            XXs[preamble:], YYs[preamble:],
            batch_size=nbatch,
            epochs=1,
            shuffle=False, verbose=1,
            callbacks=[ResetStatesCallback(reset_states_every)]
        )

        model.reset_states()
        model.evaluate(XX_validation[:preamble], YY_validation[:preamble], batch_size=nbatch, verbose=0)
        loss = model.evaluate(XX_validation[preamble:], YY_validation[preamble:], batch_size=nbatch, verbose=0)

        print "\n#### Epoch: %04d - Learning rate = %.8f, Validation loss = %08.6f" % (iii, learning_rate, loss)

        if loss < current_loss:
            
            last_checkpoint = fm.getCheckpointFile(model_name, session_timestamp, iii, loss)
            fm.saveModel(model, last_checkpoint)
            current_loss = loss
        elif last_checkpoint is not None:
            
            print "Epoch failed, loss jumped too much!"
            fm.loadModel(model, last_checkpoint)
            # Decay the learning rate.
            learning_rate *= 0.8
            KTF.set_value(model.optimizer.lr, learning_rate)

except KeyboardInterrupt:
    
    print "\n\n#### Training interrupted by user"
    filename = fm.getInterruptFile(model_name, session_timestamp, int(time.time()))
    fm.saveModel(model, filename)
    print "Current model saved to %s" % filename
