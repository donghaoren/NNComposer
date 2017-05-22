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

    seq = []
    last_item = None
    losses = []
    last_item_p = None

    for i in range(0, predict_length):
        if i < len(start):
            si = start[i]
            if i > 0:
                losses.append((
                    -np.log(np.clip(last_item_p[si[0]], 1e-6, 1 - 1e-6)),
                    np.abs(last_item[1] - si[1]),
                    np.abs(last_item[2] - si[2]),
                    np.abs(last_item[3] - si[3])
                ))
            last_item = si

        seq.append(last_item)

        x = encodedMessageToVector(last_item)
        p = model.predict(np.expand_dims(np.expand_dims(x, axis = 0), axis = 0), batch_size = 1).flatten()
        pnotes = p[3:] * 1.0
        note = sample(p[3:], diversity)
        if i < len(start):
            note = last_item[0]
        p *= 0
        p[3:] = 0
        p[3 + note] = 1
        p = model.predict(np.expand_dims(np.expand_dims(p, axis = 0), axis = 0), batch_size = 1).flatten()


        last_item = fixItem( ( note, float(p[2]), float(p[1]), float(p[0]) ) )
        last_item_p = pnotes

    return seq, np.array(losses)

somemsgs = midi2Messages(args.prefix)

random.seed(args.seed)
newmsgs, losses = sampleFromModel(model_predict, somemsgs[:args.prefixLength], args.composeLength, args.diversity)
convertEncodedMessagesToMIDI(newmsgs, args.output)