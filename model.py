from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.layers import LSTM

def build_lstm_model_512(batch_size, time_steps):
    input_size = 512
    
    model = Sequential()
    
    model.add(TimeDistributed(
        Dense(256, activation = "relu"),
        input_shape = (time_steps, input_size),
        batch_size = batch_size
    ))
              
    model.add(LSTM(256, dropout = 0.2, return_sequences = True))
    model.add(LSTM(256, dropout = 0.2, return_sequences = False))
              
    model.add(Dense(input_size))
    
    model.compile(loss = "mean_squared_error", optimizer = "adam")
              
    return model
    
    
    
build_lstm_model_512(8, 64)