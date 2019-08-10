import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Conv1D,MaxPooling1D,Flatten,RepeatVector,LSTM,TimeDistributed

from tensorflow.keras.optimizers import SGD


def build_model_cnn_lstm(train_x,train_y, n_input):
    """
    Returns trained model,provided by x_train,y_train, n_input(#of inupuys, usually 7)
    """

    verbose, epochs, batch_size = 0, 300, 1
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(Conv1D(filters=48, kernel_size=3, activation='relu',
    input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=48, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(60, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

