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

import sys
import traceback

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

def prepare_datasets(x, y, n_pre, batch_size, validation_split=0.2, loss_fn="sparse_categorical_crossentropy"):
    # Ensure x and y are DataFrames
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    
    # Ensure 'ID' is present in both x and y
    if 'ID' not in x.columns:
        x['ID'] = range(len(x))
    if 'ID' not in y.columns:
        y['ID'] = range(len(y))
    
    common_ids = sorted(set(x['ID']) & set(y['ID']))
    x = x[x['ID'].isin(common_ids)]
    y = y[y['ID'].isin(common_ids)]
    
    num_samples = len(common_ids)
    val_size = max(1, int(validation_split * num_samples))
    train_size = num_samples - val_size

    # Adjust batch_size if it's larger than the number of samples
    batch_size = min(batch_size, num_samples)

    if loss_fn == "sparse_categorical_crossentropy":
        y = y.astype(np.int32)
    else:
        y = y.astype(np.float32)
    
    y_shape = y.shape[1:]

    def dataset_generator():
        for id in common_ids:
            x_seq = x[x['ID'] == id].drop('ID', axis=1).values[-n_pre:]
            if loss_fn == "sparse_categorical_crossentropy":
                y_val = y[y['ID'] == id].drop('ID', axis=1).values[-1, 0]  # Take only the first column
            else:
                y_val = y[y['ID'] == id].drop('ID', axis=1).values[-1]
            if x_seq.shape[0] < n_pre:
                pad_width = ((n_pre - x_seq.shape[0], 0), (0, 0))
                x_seq = np.pad(x_seq, pad_width, mode='constant', constant_values=0)
            yield x_seq, y_val

    if loss_fn == "sparse_categorical_crossentropy":
        y_shape = ()
    else:
        y_shape = (y.shape[1] - 1,)  # Exclude the 'ID' column

    dataset = tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(n_pre, x.shape[1] - 1), dtype=tf.float32),
            tf.TensorSpec(shape=y_shape, dtype=tf.int32 if loss_fn == "sparse_categorical_crossentropy" else tf.float32)
        )
    )

    dataset = dataset.shuffle(buffer_size=num_samples)
    
    train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = dataset.skip(train_size).take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, train_size, val_size

def data_generator(x_data, y_data, n_pre, batch_size, loss_fn):
    x_data_grouped = x_data.groupby(level='id')
    y_data_grouped = y_data.groupby(level='id')
    
    unique_ids = sorted(set(x_data_grouped.groups.keys()) & set(y_data_grouped.groups.keys()))
    num_samples = len(unique_ids)
    
    if num_samples == 0:
        # If there are no samples, yield a dummy batch
        dummy_x = np.zeros((1, n_pre, x_data.shape[1]), dtype=np.float32)
        dummy_y = np.zeros(1, dtype=np.int32)
        yield dummy_x, dummy_y
    else:
        for i in range(0, num_samples, batch_size):
            batch_ids = unique_ids[i:min(i+batch_size, num_samples)]
            batch_x = []
            batch_y = []
            
            for id in batch_ids:
                x_group = x_data_grouped.get_group(id)
                y_group = y_data_grouped.get_group(id)
                
                x_seq = x_group.iloc[-n_pre:].values
                y_val = y_group.iloc[-1].values[0]
                
                if x_seq.shape[0] < n_pre:
                    pad_width = ((n_pre - x_seq.shape[0], 0), (0, 0))
                    x_seq = np.pad(x_seq, pad_width, mode='constant', constant_values=-1)
                
                batch_x.append(x_seq)
                batch_y.append(y_val)
            
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            
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
    lstm_1 = LSTM(int(n_hidden), dropout=dr, recurrent_dropout=dr/2, activation='tanh', recurrent_activation="sigmoid", return_sequences=True, name="LSTM_1")(masked_input)
    lstm_2 = LSTM(int(math.ceil(n_hidden/2)), dropout=dr, recurrent_dropout=dr/2, activation='tanh', recurrent_activation="sigmoid", return_sequences=False, name="LSTM_2")(lstm_1)
    
    output = Dense(output_dim, activation='softmax', name='Dense')(lstm_2)
    
    model = Model(inputs, output)
    
    optimizer = Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model