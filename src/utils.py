import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotUniform

import sys
import traceback

def get_output_signature(loss_fn, J):
    if loss_fn == "sparse_categorical_crossentropy":
        return tf.TensorSpec(shape=(None,), dtype=tf.int32)
    elif loss_fn == "binary_crossentropy":
        return tf.TensorSpec(shape=(None,), dtype=tf.float32)
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")
        
def load_data_from_csv(input_file, output_file):
    x_data = pd.read_csv(input_file)
    y_data = pd.read_csv(output_file)
    
    # Ensure 'ID' column exists and is of integer type
    if 'ID' in x_data.columns:
        x_data['ID'] = x_data['ID'].astype(int)
    elif 'id' in x_data.columns:
        x_data['ID'] = x_data['id'].astype(int)
        x_data = x_data.drop('id', axis=1)
    else:
        x_data['ID'] = range(len(x_data))

    # Handle the case where y_data is empty
    if y_data.empty:
        y_data = pd.DataFrame({'ID': x_data['ID']})
        for i in range(7):  # Assuming J=7 from the error message
            y_data[f'A{i}'] = 0
    else:
        if 'ID' in y_data.columns:
            y_data['ID'] = y_data['ID'].astype(int)
        elif 'id' in y_data.columns:
            y_data['ID'] = y_data['id'].astype(int)
            y_data = y_data.drop('id', axis=1)
        else:
            y_data['ID'] = range(len(y_data))
    
    # Merge x_data and y_data based on 'ID'
    merged_data = pd.merge(x_data, y_data, on='ID', how='inner')
    
    # Separate back into x_data and y_data
    x_columns = [col for col in merged_data.columns if col in x_data.columns]
    y_columns = [col for col in merged_data.columns if col in y_data.columns and col != 'ID']
    
    x_data = merged_data[x_columns]
    y_data = merged_data[y_columns]
    
    # Fill NaN values with -1 and ensure all data is float32
    x_data = x_data.fillna(-1).astype(np.float32)
    y_data = y_data.fillna(-1).astype(np.float32)
    
    return x_data, y_data

def data_generator(x_data, y_data, n_pre, batch_size, loss_fn, J):
    print(f"x_data shape in generator: {x_data.shape}")
    print(f"y_data shape in generator: {y_data.shape}")
    
    # Ensure 'ID' is not in x_data
    if 'ID' in x_data.columns:
        x_data = x_data.drop(columns=['ID'])
    elif 'id' in x_data.columns:
        x_data = x_data.drop(columns=['id'])
    
    print(f"x_data shape after dropping ID: {x_data.shape}")
    print(f"n_pre: {n_pre}")
    print(f"batch_size: {batch_size}")
    
    # Normalize the input data
    x_data = (x_data - x_data.mean()) / x_data.std()
    
    # Check if 'id' is in the index
    if 'id' in x_data.index.names:
        x_data_grouped = x_data.groupby(level='id')
        y_data_grouped = y_data.groupby(level='id')
    # If not, create a default id
    else:
        print("Warning: 'id' not found in index. Creating default id.")
        x_data = x_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)
        x_data['id'] = range(len(x_data))
        y_data['id'] = range(len(y_data))
        x_data = x_data.set_index('id')
        y_data = y_data.set_index('id')
        x_data_grouped = x_data.groupby(level='id')
        y_data_grouped = y_data.groupby(level='id')
    
    unique_ids = sorted(set(x_data_grouped.groups.keys()) & set(y_data_grouped.groups.keys()))
    num_samples = len(unique_ids)
    
    for i in range(0, num_samples, batch_size):
        batch_ids = unique_ids[i:min(i+batch_size, num_samples)]
        batch_x = []
        batch_y = []
        
        for id in batch_ids:
            x_group = x_data_grouped.get_group(id)
            y_group = y_data_grouped.get_group(id)
            
            x_seq = x_group.iloc[-n_pre:].values
            if loss_fn == "sparse_categorical_crossentropy":
                y_val = y_group.iloc[-1].values[0]  # Single integer label
            elif loss_fn == "binary_crossentropy":
                y_val = y_group.iloc[-1].values[0]  # Use only the first column for binary classification
            else:
                raise ValueError(f"Unsupported loss function: {loss_fn}")
            
            if x_seq.shape[0] < n_pre:
                pad_width = ((n_pre - x_seq.shape[0], 0), (0, 0))
                x_seq = np.pad(x_seq, pad_width, mode='constant', constant_values=-1)
            
            batch_x.append(x_seq)
            batch_y.append(y_val)
        
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        
        if loss_fn == "binary_crossentropy":
            batch_y = (batch_y > 0).astype(int)  # Ensure binary labels
        elif loss_fn == "sparse_categorical_crossentropy":
            batch_y = batch_y % J  # Ensure labels are within [0, J-1]
        
        print(f"batch_x shape: {batch_x.shape}")
        print(f"batch_y shape: {batch_y.shape}")
        print(f"batch_y unique values: {np.unique(batch_y)}")
        
        yield batch_x, batch_y

def load_data(file_path, is_output=False):
    data = pd.read_csv(file_path)
    if is_output:
        data = data.pivot(index=['id', 't'], columns='feature', values='value')
        data = data.astype(int)
    else:
        data = data.pivot(index=['id', 't'], columns='feature', values='value').fillna(-1)
    
    # Ensure all IDs have the same number of time steps
    max_t = data.index.get_level_values('t').max()
    full_index = pd.MultiIndex.from_product([data.index.get_level_values('id').unique(), range(1, max_t+1)],
                                            names=['id', 't'])
    data = data.reindex(full_index, fill_value=-1)
    
    return data

def create_model(input_shape, output_dim, lr, dr, n_hidden, hidden_activation, out_activation, loss_fn, J):
    inputs = Input(shape=input_shape, name="Inputs")
    masked_input = Masking(mask_value=0.0)(inputs)
    lstm_1 = LSTM(int(n_hidden), dropout=dr, recurrent_dropout=dr/2, activation='tanh', recurrent_activation="sigmoid", return_sequences=True, name="LSTM_1", kernel_regularizer=l2(0.01), kernel_initializer=GlorotUniform())(masked_input)
    lstm_2 = LSTM(int(math.ceil(n_hidden/2)), dropout=dr, recurrent_dropout=dr/2, activation='tanh', recurrent_activation="sigmoid", return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.01), kernel_initializer=GlorotUniform())(lstm_1)
    
    if loss_fn == "sparse_categorical_crossentropy":
        output = Dense(J, activation='softmax', name='Dense', kernel_regularizer=l2(0.01), kernel_initializer=GlorotUniform())(lstm_2)
    elif loss_fn == "binary_crossentropy":
        output = Dense(1, activation='sigmoid', name='Dense', kernel_regularizer=l2(0.01), kernel_initializer=GlorotUniform())(lstm_2)
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")
    
    model = Model(inputs, output)
    
    optimizer = Adam(learning_rate=lr, clipnorm=1.0, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model