from __future__ import print_function

import os.path
from os import path

import sys
import math
import numpy as np
import pandas as pd

import keras
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import LSTM, Input, Dense
from keras.callbacks import CSVLogger, EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam

# Select gpu
import os
if gpu < 3:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(gpu)

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

def custom_loss(y_true, y_pred):
    y_true_reshaped = tf.reshape(y_true, [-1, 1])
    y_pred_reshaped = tf.reshape(y_pred, [-1, 7]) # The model's output is already shaped as (batch_size, 10000, 7)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_reshaped, y_pred_reshaped)
    return tf.reduce_mean(loss)

def create_model(n_pre, nb_features, output_dim, lr, dr, n_hidden, hidden_activation, out_activation, loss_fn):
    """ 
        creates, compiles and returns a RNN model 
        @param nb_features: the number of features in the model
    """
    # Define model parameters

    inputs = Input(batch_shape=(int(nb_batches), n_pre, nb_features), name="Inputs")
    lstm_1 = LSTM(int(n_hidden), dropout=dr, activation= hidden_activation, recurrent_activation="sigmoid", return_sequences=True, stateful=True, name="LSTM_1")(inputs) 
    lstm_2 = LSTM(int(math.ceil(n_hidden/2)), dropout=dr, activation= hidden_activation, recurrent_activation="sigmoid", return_sequences=False, stateful=True, name="LSTM_2")(lstm_1) 
    
    if loss_fn=="sparse_categorical_crossentropy":
        output = Dense(10000 * 7, activation='linear', name='Dense')(lstm_2)
        output_reshaped = tf.keras.layers.Reshape((10000, 7))(output)
        softmax_output = tf.keras.layers.Activation('softmax')(output_reshaped)
    else:
        output= Dense(output_dim, activation=out_activation, name='Dense')(lstm_2)


    if loss_fn=="sparse_categorical_crossentropy":
        model = Model(inputs, softmax_output) 
    else:
        model = Model(inputs, output)

    # Compile
    if loss_fn=="sparse_categorical_crossentropy":
        model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss)
    else:
        model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn)
    
    return model

def train_model(model, dataX, dataY, epoch_count, batches):

    # Prepare model checkpoints and callbacks

    stopping = EarlyStopping(monitor='val_loss', patience=int(patience), min_delta=0, verbose=1, mode='min', restore_best_weights=True)

    terminate = TerminateOnNaN()

    # Model fit

    for epoch in range(epoch_count):
        history = model.fit(x=dataX, 
            y=dataY, 
            batch_size=batches, 
            verbose=1,
            epochs=1,  # Fit one epoch at a time
            callbacks=[stopping,terminate],
            validation_split=0.2,
            shuffle=False)
        
        model.reset_states()  # Reset the states at the end of each epoch

def test_model():

    n_post = int(1)
    n_pre = int(window_size)
    seq_len = int(t_end)

    x = np.array(pd.read_csv("{}input_data.csv".format(output_dir)))
    y = np.array(pd.read_csv("{}output_data.csv".format(output_dir)))

    print('raw x shape', x.shape)   
    print('raw y shape', y.shape) 

    dX, dY = [], []
    for i in range(seq_len-n_pre-n_post):
        dX.append(x[i:i+n_pre])
        dY.append(y[i+n_pre])
    
    dataX = np.array(dX)
    dataY = np.array(dY)

    print('dataX shape:', dataX.shape)
    print('dataY shape:', dataY.shape)

    if loss_fn=="sparse_categorical_crossentropy":
        dataY = dataY-1

    nb_features = dataX.shape[2]
    output_dim = dataY.shape[1]

    print('nb_features:', nb_features)
    print('output_dim:', output_dim)
  
    # create and fit the LSTM network
    print('creating model...')
    model = create_model(n_pre, nb_features, output_dim, lr, dr, n_hidden, hidden_activation, out_activation, loss_fn)

    train_model(model, dataX, dataY, int(epochs), int(nb_batches))

    # now test

    print('Generate predictions')

    preds_test = model.predict(dataX, batch_size=int(nb_batches), verbose=0)

    print('predictions shape =', preds_test.shape)

    # Save predictions

    print('Saving to {}lstm_preds.npy'.format(output_dir))

    np.save("{}lstm_preds.npy".format(output_dir), preds_test)

def main():
    test_model()
    return 1

if __name__ == "__main__":
	main()