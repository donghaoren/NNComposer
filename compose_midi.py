import argparse

parser = argparse.ArgumentParser(description='Create a dataset from MIDI files')

parser.add_argument('modelName', type=unicode, help="the name of the model to use")
parser.add_argument('modelFile', type=unicode, help="the model to use for the composition")
parser.add_argument('prefix', type=unicode, help="a MIDI file to initiate the composition")
parser.add_argument('output', type=unicode, help="the MIDI file to output to")
parser.add_argument('--prefixLength', type=int, help="the number of events for the prefix sequence", default=100)
parser.add_argument('--composeLength', type=int, help="the number of events for the composition", default=5000)
parser.add_argument('--diversity', type=float, help="the diversity number to use", default=1.0)
parser.add_argument('--seed', type=int, help="the random seed to use for the composition", default=0)
parser.add_argument('--device', type=unicode, help="the Tensorflow device to use", default="/cpu:0")

args = parser.parse_args()

from utils.gpu_config import tfSessionAllowGrowth
tfSessionAllowGrowth()

import numpy as np
import tensorflow as tf
import random

from midi_models.models import buildModel
from midi_signals.midi import midi2Messages
from midi_signals.midi import encodedMessageToVector, vectorToEncodedMessage, convertEncodedMessagesToMIDI, sample

with tf.device(args.device):
    model_predict = buildModel(args.modelName, 1, 1)

model_predict.load_weights(args.modelFile)

def fixItem(item):
    if item[3] > 1:
        return item[:3] + (1,)
    return item

def sampleFromModel(model, start, predict_length = 4000, diversity = 1.0):
    model.reset_states()
    
    last_note = start[0][0]

    seq = []

    for i in range(1, predict_length):
        x = np.zeros((180,))
        x[3 + last_note] = 1
        p = model.predict(np.expand_dims(np.expand_dims(x, axis = 0), axis = 0), batch_size = 1).flatten()
        # p[0:3] is predicted last_note's velocity.
        x[0:3] = p[0:3]
        
        if i - 1 < len(start):
            x[2] = start[i - 1][1]
            x[1] = start[i - 1][2]
            x[0] = start[i - 1][3]
        
        last_item = ( last_note, float(x[2]), float(x[1]), float(x[0]) )
        last_item = fixItem(last_item)
        
        seq.append(last_item)
        
        p = model.predict(np.expand_dims(np.expand_dims(x, axis = 0), axis = 0), batch_size = 1).flatten()
        last_note = sample(p[3:], diversity)
        
        if i < len(start):
            last_note = start[i][0]

    return seq

somemsgs = midi2Messages(args.prefix)

np.random.seed(args.seed)
newmsgs = sampleFromModel(model_predict, somemsgs[:args.prefixLength], args.composeLength, args.diversity)
convertEncodedMessagesToMIDI(newmsgs, args.output)