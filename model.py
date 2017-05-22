from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import math


def build_lstm_model_512(batch_size, time_steps):
    input_size = 512
    
    model = Sequential()
    
    model.add(TimeDistributed(
        Dense(1024, activation = "relu"),
        input_shape = (time_steps, input_size),
        batch_size = batch_size
    ))
              
    model.add(LSTM(1024, return_sequences = True))
    model.add(Dropout(0.1))

    model.add(LSTM(1024, return_sequences = True))
    model.add(Dropout(0.1))
              
    model.add(TimeDistributed(Dense(input_size)))
    
    model.compile(loss = "mean_squared_error", optimizer = "adam")
              
    return model

def build_midi_model(batch_size, time_steps):
    
    length = 512
    
    def separate_softmax(x):
        return tf.concat([tf.nn.softmax(x[:,:,:88*2]), tf.nn.softmax(x[:,:,88*2:])], 2)
    
    def loss(y_true, y_pred):
        epsilon = 1e-8
        return -tf.reduce_sum(y_true * tf.log(tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)))
    
    model = Sequential()
    model.add(GRU(256, stateful=True, return_sequences=True, batch_size=batch_size, input_shape=(time_steps, length)))
    model.add(Dropout(0.1))
    model.add(GRU(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(length, activation=separate_softmax))

    model.compile(loss=loss, optimizer='adam')
    
    return model

def build_midi_model_180(batch_size, time_steps):
    
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
        part_softmax = -tf.reduce_sum(y_true[:,:,3:] * tf.log(tf.clip_by_value(y_pred[:,:,3:], epsilon, 1.0 - epsilon)), axis = -1)
        part_dt = gaussian_loss(y_true[:,:,0:1], y_pred[:,:,0:1])
        part_velocity = gaussian_loss(y_true[:,:,1:2], y_pred[:,:,1:2])
        part_pedal = gaussian_loss(y_true[:,:,2:3], y_pred[:,:,2:3])
        return part_softmax + part_dt + part_velocity + part_pedal
    
    model = Sequential()
    model.add(TimeDistributed(
        Dense(256, activation="elu"),
        batch_size=batch_size, input_shape=(time_steps, length)
    ))
    model.add(Dropout(0.1))
    model.add(GRU(512, stateful=True, return_sequences=True)) #, batch_size=batch_size, input_shape=(time_steps, length)))
    model.add(Dropout(0.1))
    model.add(GRU(512, stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(length, activation=activation))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model

def build_midi_model_180_large(batch_size, time_steps):
    
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
        part_softmax = -tf.reduce_sum(y_true[:,:,3:] * tf.log(tf.clip_by_value(y_pred[:,:,3:], epsilon, 1.0 - epsilon)), axis = -1)
        part_dt = gaussian_loss(y_true[:,:,0:1], y_pred[:,:,0:1])
        part_velocity = gaussian_loss(y_true[:,:,1:2], y_pred[:,:,1:2])
        part_pedal = gaussian_loss(y_true[:,:,2:3], y_pred[:,:,2:3])
        return part_softmax + part_dt + part_velocity + part_pedal
    
    model = Sequential()
    model.add(TimeDistributed(
        Dense(1024, activation="elu"),
        batch_size=batch_size, input_shape=(time_steps, length)
    ))
    model.add(Dropout(0.1))
    model.add(GRU(1024, stateful=True, return_sequences=True)) #, batch_size=batch_size, input_shape=(time_steps, length)))
    model.add(Dropout(0.1))
    model.add(GRU(1024, stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(length, activation=activation))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model

def build_midi_model_180_deep(batch_size, time_steps):
    
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
        part_softmax = -tf.reduce_sum(y_true[:,:,3:] * tf.log(tf.clip_by_value(y_pred[:,:,3:], epsilon, 1.0 - epsilon)), axis = -1)
        part_dt = gaussian_loss(y_true[:,:,0:1], y_pred[:,:,0:1])
        part_velocity = gaussian_loss(y_true[:,:,1:2], y_pred[:,:,1:2])
        part_pedal = gaussian_loss(y_true[:,:,2:3], y_pred[:,:,2:3])
        return part_softmax + part_dt + part_velocity + part_pedal
    
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

def build_midi_model_180_deep_doubleinput(batch_size, time_steps):
    
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

def build_midi_model_180_deep2_doubleinput(batch_size, time_steps):
    
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
    model.add(Dense(1024, activation="elu"))
    model.add(Dense(length, activation=activation))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model

def build_midi_model_180_gauss(batch_size, time_steps):
    
    length = 180
    
    def activation(x):
        return tf.concat([
            tf.nn.elu(x[:,:,0:1]),
            tf.nn.sigmoid(x[:,:,1:2]),
            tf.nn.elu(x[:,:,2:3]),
            tf.nn.sigmoid(x[:,:,3:4]),
            tf.nn.elu(x[:,:,4:5]),
            tf.nn.sigmoid(x[:,:,5:6]),
            tf.nn.softmax(x[:,:,6:])
        ], 2)

    def gaussian_loss(y_true, y_pred):
        sigma = tf.maximum(y_pred[:,:,1], 1e-4)
        sqr_part = tf.square(y_true[:,:,0] - y_pred[:,:,0]) / (tf.square(sigma) * 2)
        return sqr_part + tf.log(sigma) + math.log(math.sqrt(math.pi * 2))


    def loss(y_true, y_pred):
        epsilon = 1e-6
        part_softmax = -tf.reduce_sum(y_true[:,:,3:] * tf.log(tf.clip_by_value(y_pred[:,:,6:], epsilon, 1.0 - epsilon)), axis = -1)
        part_dt = gaussian_loss(y_true[:,:,0:1], y_pred[:,:,0:2])
        part_velocity = gaussian_loss(y_true[:,:,1:2], y_pred[:,:,2:4])
        part_pedal = gaussian_loss(y_true[:,:,2:3], y_pred[:,:,4:6])
        return part_softmax * 4 + part_dt + part_velocity + part_pedal
    
    model = Sequential()
    model.add(TimeDistributed(
        Dense(256, activation="elu"),
        batch_size=batch_size, input_shape=(time_steps, length)
    ))
    model.add(Dropout(0.1))
    model.add(GRU(512, stateful=True, return_sequences=True)) #, batch_size=batch_size, input_shape=(time_steps, length)))
    model.add(Dropout(0.1))
    model.add(GRU(512, stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(length + 3, activation=activation))

    optimizer = Adam(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model
