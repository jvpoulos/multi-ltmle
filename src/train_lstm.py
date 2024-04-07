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
from keras.layers import LSTM, Input, Dense, Masking
from keras.callbacks import CSVLogger, EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda

def create_model(input_shape, output_dim, lr, dr, n_hidden, hidden_activation, out_activation, loss_fn, J=int(7)):
    """ 
        creates, compiles and returns a RNN model 
        @param input_shape: the shape of the input data
    """
    print("input_shape:", input_shape)
    print("output_dim:", output_dim)
    print("J:", J)

    # Define model parameters
    inputs = Input(shape=input_shape, name="Inputs") 
    masked_input = Masking(mask_value=-1.0)(inputs)
    lstm_1 = LSTM(int(n_hidden), dropout=dr, activation=hidden_activation, recurrent_activation="sigmoid", return_sequences=True, name="LSTM_1")(masked_input) 
    lstm_2 = LSTM(int(math.ceil(n_hidden/2)), dropout=dr, activation=hidden_activation, recurrent_activation="sigmoid", return_sequences=True, name="LSTM_2")(lstm_1) 

    output = Dense(int(J), activation='linear', name='Dense')(lstm_2)

    model = Model(inputs, output)

    # Compile
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn)
    
    return model

def train_model(model, dataX, dataY, epoch_count, batches):

    # Prepare model checkpoints and callbacks

    stopping = EarlyStopping(monitor='val_loss', patience=int(patience), min_delta=0, verbose=1, mode='min', restore_best_weights=True)

    terminate = TerminateOnNaN()

    # Model fit
    history = model.fit(x=dataX, 
        y=dataY, 
        batch_size=batches, 
        verbose=1,
        epochs=epoch_count, 
        callbacks=[stopping,terminate],
        validation_split=0.2,
        shuffle=False)

def test_model():
    n_pre = int(window_size)
        
    if loss_fn == "sparse_categorical_crossentropy":
        input_data = pd.read_csv("{}input_cat_data.csv".format(output_dir), low_memory=False)
    else:
        input_data = pd.read_csv("{}input_bin_data.csv".format(output_dir), low_memory=False)
    time_step_col = 't'
    id_col = 'id'
    feature_cols = 'feature'
    x = input_data.pivot_table(index=[id_col, time_step_col], columns=feature_cols, values='value', aggfunc='first').values

    output_data = pd.read_csv("{}output_cat_data.csv".format(output_dir), low_memory=False)
    output_col = output_data.columns[2]  # Assuming the third column is the output
    y = output_data.pivot_table(index=[id_col, time_step_col], columns=output_col, aggfunc='first').values

    print("Unique values in y:", np.unique(y))

    print('raw x shape', x.shape)   
    print('raw y shape', y.shape)

    num_individuals = x.shape[0]
    num_timesteps = x.shape[1]
    num_features = x.shape[2] if len(x.shape) == 3 else 1

    dataX = np.zeros((num_individuals, num_timesteps - n_pre + 1, n_pre, 1))
    dataY = np.zeros((num_individuals, num_timesteps - n_pre + 1, y.shape[1]))

    for i in range(num_individuals):
        for j in range(num_timesteps - n_pre + 1):
            if len(x.shape) == 3:
                dataX[i, j, :, 0] = x[i, j:j+n_pre, 0]
            else:
                dataX[i, j, :, 0] = x[i, j:j+n_pre]
            dataY[i, j, :] = y[j+n_pre-1, :]

    print('dataX shape:', dataX.shape)
    print('dataY shape:', dataY.shape)

    input_shape = (n_pre, 1)
    print('input_shape:', input_shape)

    if loss_fn == "sparse_categorical_crossentropy":
        print("dataY before conversion:")
        print(dataY)
        print("Minimum value in dataY:", np.min(dataY))
        print("Maximum value in dataY:", np.max(dataY))
        print("Unique values in dataY:", np.unique(dataY))
        
        # Convert dataY to integer type
        dataY = dataY.astype(np.int32)
        
        print("dataY after conversion:")
        print(dataY)
        print("Minimum value in dataY:", np.min(dataY))
        print("Maximum value in dataY:", np.max(dataY))
        print("Unique values in dataY:", np.unique(dataY))
        
        # Clip the values in dataY to be within the range of 1 to 7
        dataY = np.clip(dataY, 1, 7)
        
        print("dataY after clipping:")
        print(dataY)
        print("Minimum value in dataY:", np.min(dataY))
        print("Maximum value in dataY:", np.max(dataY))
        print("Unique values in dataY:", np.unique(dataY))

    nb_features = x.shape[1]
    output_dim = y.shape[1]

    print('nb_features:', nb_features)
    print('output_dim:', output_dim)

    input_shape = dataX.shape[1:]
    print('input_shape:', input_shape)
  
    # create and fit the LSTM network
    print('creating model...')
    model = create_model(input_shape, output_dim, lr, dr, n_hidden, hidden_activation, out_activation, loss_fn, J)

    train_model(model, dataX, dataY, int(epochs), int(nb_batches))

    # Save the trained model
    if loss_fn=="sparse_categorical_crossentropy":
        model_path = os.path.join(output_dir, 'trained_cat_model.h5')
    else:
        model_path = os.path.join(output_dir, 'trained_bin_model.h5')

    print('Saving model to {}'.format(model_path))
    model.save(model_path)

    # now test

    print('Generate predictions')

    logits_test = model.predict(dataX, batch_size=int(nb_batches), verbose=0)
    preds_test = np.argmax(logits_test, axis=-1)

    print('predictions shape =', preds_test.shape)

    # Save predictions

    if loss_fn=="sparse_categorical_crossentropy":
        print('Saving to {}lstm_preds.npy'.format(output_dir))
        np.save("{}lstm_cat_preds.npy".format(output_dir), preds_test)
    else:
        print('Saving to {}lstm_preds.npy'.format(output_dir))
        np.save("{}lstm_bin_preds.npy".format(output_dir), preds_test)

def main():
    test_model()
    return 1

if __name__ == "__main__":
    main()