from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, Conv1D, Activation, TimeDistributed, Dropout
from keras.layers import LSTM, GRU, concatenate
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import math

from models import defineMIDIModel

@defineMIDIModel("Ex_Convolutional", "double_slice")
def Ex_Convolutional(batch_size, time_steps):
    
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
    
    inputs = Input(batch_shape=(batch_size, time_steps, length))
    
    # Split the inputs into 3
    velocity = Lambda(lambda x: x[:,:,0:3])(inputs)
    pedal = Lambda(lambda x: x[:,:,3:4])(inputs)
    note = Lambda(lambda x: tf.expand_dims(x[:,:,4:180], axis = 3))(inputs)
    
    note_conv = TimeDistributed(
        Conv1D(128, 48, activation="relu", padding = "same")
    )(note)
    
    note_conv = TimeDistributed(
        Conv1D(64, 24, activation="relu", padding = "same")
    )(note_conv)
    
    note_conv = Lambda(lambda x: tf.reshape(x, (batch_size, time_steps, 176 * 64)))(note_conv)
    
    y = concatenate([ velocity, pedal, note_conv ])
    
    y = GRU(1024, stateful=True, return_sequences=True)(y)
    y = Dropout(0.2)(y)
    y = Dense(length, activation=activation)(y)
                        
    model = Model(inputs = inputs, outputs = y)

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model
