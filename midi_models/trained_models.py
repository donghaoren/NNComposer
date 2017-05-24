from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, Conv1D, Activation, TimeDistributed, Dropout
from keras.layers import LSTM, GRU, concatenate
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import math

from models import defineMIDIModel

@defineMIDIModel("D1024_G1024_G256_G1024_D180", "double_slice")
def D1024_G1024_G256_G1024_D180(batch_size, time_steps):
    length = 180
    def activation(x):
        return tf.concat([
            tf.nn.elu(x[:,:,0:3]),
            tf.nn.softmax(x[:,:,3:])
        ], 2)

    def gaussian_loss(y_true, y_pred):
        sqr_part = tf.square(y_true[:,:,0] - y_pred[:,:,0])
        return sqr_part


    def loss(y_true, y_pred):
        epsilon = 1e-6
        part_softmax = -tf.reduce_sum(y_true[:,:,3:180] * tf.log(tf.clip_by_value(y_pred[:,:,3:], epsilon, 1.0 - epsilon)), axis = -1)
        part_dt = gaussian_loss(y_true[:,:,0:1], y_pred[:,:,0:1])
        part_velocity = gaussian_loss(y_true[:,:,1:2], y_pred[:,:,1:2])
        part_pedal = gaussian_loss(y_true[:,:,2:3], y_pred[:,:,2:3])
        return part_softmax + (part_dt + part_velocity + part_pedal) * y_true[:,:,180]
    
    model = Sequential()
    model.add(TimeDistributed(
        Dense(1024, activation="elu"),
        batch_size=batch_size, input_shape=(time_steps, length)
    ))
    model.add(GRU(1024, stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(1024, stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(length, activation=activation))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model

@defineMIDIModel("D1024_L1024_L256_L1024_D180", "double_slice")
def D1024_L1024_L256_L1024_D180(batch_size, time_steps):
    
    length = 180
    
    def activation(x):
        return tf.concat([
            tf.nn.elu(x[:,:,0:3]),
            tf.nn.softmax(x[:,:,3:])
        ], 2)

    def gaussian_loss(y_true, y_pred):
        sqr_part = tf.square(y_true[:,:,0] - y_pred[:,:,0])
        return sqr_part


    def loss(y_true, y_pred):
        epsilon = 1e-6
        part_softmax = -tf.reduce_sum(y_true[:,:,3:180] * tf.log(tf.clip_by_value(y_pred[:,:,3:], epsilon, 1.0 - epsilon)), axis = -1)
        part_dt = gaussian_loss(y_true[:,:,0:1], y_pred[:,:,0:1])
        part_velocity = gaussian_loss(y_true[:,:,1:2], y_pred[:,:,1:2])
        part_pedal = gaussian_loss(y_true[:,:,2:3], y_pred[:,:,2:3])
        return part_softmax + (part_dt + part_velocity + part_pedal) * y_true[:,:,180]
    
    model = Sequential()
    model.add(TimeDistributed(
        Dense(1024, activation="elu"),
        batch_size=batch_size, input_shape=(time_steps, length)
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(1024, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(1024, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(length, activation=activation))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model