from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, Conv1D, Activation, TimeDistributed, Dropout
from keras.layers import LSTM, GRU, concatenate
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import math

models = {}

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

@defineMIDIModel("Single_Layer5_1024_256_1024", "single_slice")
def Single_Layer5_1024_256_1024(batch_size, time_steps):
    
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

@defineMIDIModel("Layer5_1024_256_1024", "double_slice")
def Layer5_1024_256_1024(batch_size, time_steps):
    
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

@defineMIDIModel("Layer5_1024_256_1024_LSTM", "double_slice")
def Layer5_1024_256_1024_LSTM(batch_size, time_steps):
    
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

@defineMIDIModel("Layer5_1024_256_1024_GRU", "double_slice")
def Layer5_1024_256_1024_GRU(batch_size, time_steps):
    
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
    model.add(GRU(1024, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(1024, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(length, activation=activation))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model

@defineMIDIModel("Layer5_512_512_512_LSTM", "double_slice")
def Layer5_512_512_512_LSTM(batch_size, time_steps):
    
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
        Dense(512, activation="elu"),
        batch_size=batch_size, input_shape=(time_steps, length)
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(512, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(length, activation=activation))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model

@defineMIDIModel("Layer5_256x4_LSTM", "double_slice")
def Layer5_512x4_LSTM(batch_size, time_steps):
    
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
        Dense(256, activation="elu"),
        batch_size=batch_size, input_shape=(time_steps, length)
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(length, activation=activation))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model

@defineMIDIModel("Layer5_Conv", "double_slice")
def Layer5_Conv(batch_size, time_steps):
    
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
    
    note_conv = Dropout(0.2)(note_conv)
    
    note_conv = TimeDistributed(
        Conv1D(64, 24, activation="relu", padding = "same")
    )(note_conv)
    
    note_conv = Dropout(0.2)(note_conv)
    
    note_conv = Lambda(lambda x: tf.reshape(x, (batch_size, time_steps, 176 * 64)))(note_conv)
    
    y = concatenate([ velocity, pedal, note_conv ])
    
    # y = TimeDistributed(Dense(256, activation="relu"))(y)
    # y = GRU(256, stateful=True, return_sequences=True)(y)
    # y = Dropout(0.1)(y)
    # y = GRU(64, stateful=True, return_sequences=True)(y)
    # y = Dropout(0.1)(y)
    y = GRU(256, stateful=True, return_sequences=True)(y)
    y = Dropout(0.2)(y)
    y = GRU(256, stateful=True, return_sequences=True)(y)
    y = Dropout(0.2)(y)
    y = GRU(1024, stateful=True, return_sequences=True)(y)
    y = Dropout(0.2)(y)
    y = Dense(length, activation=activation)(y)
                        
    model = Model(inputs = inputs, outputs = y)

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


@defineMIDIModel("Layer5_256_64_256", "double_slice")
def Layer5_256_64_256(batch_size, time_steps):
    
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
        Dense(256, activation="elu"),
        batch_size=batch_size, input_shape=(time_steps, length)
    ))
    model.add(GRU(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(64, stateful=True, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(length, activation=activation))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


@defineMIDIModel("TestModel", "double_slice")
def TestModel(batch_size, time_steps):
    
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
    model.add(GRU(1024, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(256, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(1024, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(length))

    optimizer = RMSprop(lr = 0.001)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model