from  tensorflow  import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM,Dense,Conv1D,MaxPooling1D,Flatten,RepeatVector,LSTM,TimeDistributed

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.callbacks import Callback, ModelCheckpoint


def build_model_cnn_lstm(train_x,train_y, config=[5,48,60], n_input=7,filepath='model.h5'):
    """
    Returns trained model,provided by x_train,y_train, n_input(#of inupuys, usually 7)
    """
    
#     checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#     callbacks_list = [checkpoint]

    n_epoch ,cnn_filters, dense_neur = config[0],config[1],config[2]
    verbose, epochs, batch_size = 0, n_epoch, 20
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
#     y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    
    # define model
    model = Sequential()
    model.add(Conv1D(filters=cnn_filters, kernel_size=3, activation='relu',
    input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=cnn_filters, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(300, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(dense_neur, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y,epochs=epochs, batch_size=batch_size, shuffle = False,verbose=verbose)
    return model


