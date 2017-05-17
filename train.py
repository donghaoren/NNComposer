from datasets.generate import generateDataset
from audio_signals.stft import STFTTransform
import numpy as np
import math
import os
import random
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session():
    gpu_options = tf.GPUOptions(allow_growth = True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

from model import build_lstm_model_512

batch_size = 64
time_steps = 32
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

# Initialize from Epoch: 60
resume_epoch = 0
if resume_epoch > 0:
    model.load_weights("models/epoch-%06d.h5" % resume_epoch)

print "Loading training datasets into memory..."

chunk_length = 180 * 64 # 11520 ~ 33.43 seconds
chunk_step = 180

datas = []
for filename in files:
    data = np.load(filename)
    data = np.log(np.maximum(1e-6, data)) - math.log(1e-6)
    bins = data.shape[1]
    datas.append(data)
        
print "Start training..."

for epoch in range(resume_epoch + 1, 10000):
    print "####### Epoch %06d #######" % (epoch)
    X = np.zeros((chunk_length / time_steps * batch_size, time_steps, bins))
    Y = np.zeros((chunk_length / time_steps * batch_size, time_steps, bins))
    for i in range(batch_size):
        while True:
            data = random.choice(datas)
            if data.shape[0] - chunk_length - 2 < 1: continue
            s = random.randint(0, data.shape[0] - chunk_length - 2)
            break
            
        for j in range(chunk_length / time_steps):
            X[i + j * batch_size,:,:] = data[s:s+time_steps,:]
            Y[i + j * batch_size,:,:] = data[s+1:s+1+time_steps,:]
            s += time_steps

    model.reset_states()
    model.fit(X, Y, epochs=1, shuffle=False, batch_size=batch_size)

    model.save("models/epoch-%06d.h5" % epoch)
