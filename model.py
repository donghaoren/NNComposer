from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed, Dropout
from keras.layers import LSTM, GRU
import tensorflow as tf

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