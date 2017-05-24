from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, Conv1D, Activation, TimeDistributed, Dropout
from keras.layers import LSTM, GRU, concatenate
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import math

models = {}

# Input format:

# double_slice:
#     X       Y
# N(x0)   T(x0)
# T(x0)   N(x1)
# N(x1)   T(x1)
# T(x1)   N(x2)
# N(x2)   T(x2)
# T(x2)   N(x3)
# N(x3)   T(x3)
# T(x3)   N(x4)
#   ...     ...
#
# N(x0) is the note information of event X0
# T(x0) is the note information of event X0, plus timing and velocity information

def defineMIDIModel(name, input_format):
    def decorator(f):
        models[name] = {
            "name": name,
            "build": f,
            "input_format": input_format
        }
    return decorator

def buildModel(name, batch_size, time_steps):
    return models[name]["build"](batch_size, time_steps)
    
def getModelProperties(name):
    return {
        "input_format": models[name]["input_format"]
    }

import trained_models
import experiments
import lstm_vs_gru
import convolutional