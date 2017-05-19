import numpy as np
from model import build_midi_model_180_large
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import Callback
import time

def get_session():
    gpu_options = tf.GPUOptions(allow_growth = True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

length = 180

print("Build model...")
    
with tf.device("/gpu:1"):
    nbatch = 64
    maxlen = 64
    model = build_midi_model_180_large(nbatch, maxlen)

print('Loading data...')

import pickle
with open("data/midi_piano_rolls_dataset.pkl", "rb") as f:
    data = pickle.load(f)

from midi_signals.midi import encodedMessageToVector
    
print('Vectorization...')
X = np.zeros((len(data), length), dtype=np.float32)
for i in range(0, len(data)):
    X[i,:] = encodedMessageToVector(data[i])
    
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

XX_validation, YY_validation = generateXYChunk(X, nbatch, 8192, maxlen, length)

current_loss = 1e10
last_checkpoint = None

print('Training...')

file_prefix = "%d" % (int(time.time()))

class ResetStatesCallback(Callback):
    def __init__(self, max_len):
        self.counter = 0
        self.max_len = max_len

    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.max_len == 0:
            self.model.reset_states()
        self.counter += 1

learning_rate = 0.0001
preamble = 16 * nbatch # feed through the first # of samples before training/testing

load_checkpoint = "restart.h5"
# load_checkpoint = None

if load_checkpoint is not None:
    model.load_weights(load_checkpoint)
    last_checkpoint = load_checkpoint
    model.reset_states()
    model.evaluate(XX_validation[:preamble], YY_validation[:preamble], batch_size=nbatch, verbose=0)
    current_loss = model.evaluate(XX_validation[preamble:], YY_validation[preamble:], batch_size=nbatch, verbose=0)
    print "Loaded weights from %s: Learning rate = %.8f, Validation loss = %08.6f" % (load_checkpoint, learning_rate, current_loss)

KTF.set_value(model.optimizer.lr, learning_rate)



try:
    for iii in range(0, 100000):
        # train_length = min((iii + 1)  * nbatch, 65536)
        train_length = 65536
        
        for repeat in range(max(1, 65536 / train_length)):
            XXs, YYs = generateXYChunk(X, nbatch, train_length, maxlen, length)
            
            model.reset_states()
            model.evaluate(XXs[:preamble], YYs[:preamble], batch_size=nbatch, verbose=0)
            model.fit(XXs[preamble:], YYs[preamble:], batch_size=nbatch, epochs=1, shuffle=False, verbose=1, callbacks = [ResetStatesCallback(train_length)])

        model.reset_states()
        model.evaluate(XX_validation[:preamble], YY_validation[:preamble], batch_size=nbatch, verbose=0)
        loss = model.evaluate(XX_validation[preamble:], YY_validation[preamble:], batch_size=nbatch, verbose=0)

        print "\n#### Epoch: %04d - Learning rate = %.8f, Validation loss = %08.6f" % (iii, learning_rate, loss)

        if loss < current_loss:
            last_checkpoint = "midi_models_180_large/%s-midi-%06d-%08.6f.h5" % (file_prefix, iii, loss)
            model.save(last_checkpoint)
            current_loss = loss
        elif last_checkpoint is not None:
            print "Epoch failed, loss jumped too much!"
            model.load_weights(last_checkpoint)
            # Decay the learning rate.
            learning_rate *= 0.8
            KTF.set_value(model.optimizer.lr, learning_rate)

except KeyboardInterrupt:
    print "\n\n#### Training interrupted by user"
    model.save(file_prefix + "-interrupted.h5")
    print "Current model saved to train-interrupted.h5"
