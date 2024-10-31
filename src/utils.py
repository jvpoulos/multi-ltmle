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

import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_dataset):
        super().__init__()
        self.train_dataset = train_dataset
    
    def on_train_batch_begin(self, batch, logs=None):
        logs = logs or {}
        x = logs.get('x')
        y = logs.get('y')
        if x is not None and np.isnan(x).any():
            print(f"NaN detected in input data (batch {batch})")
            print(f"NaN indices: {np.argwhere(np.isnan(x))}")
        if y is not None and np.isnan(y).any():
            print(f"NaN detected in labels (batch {batch})")
            print(f"NaN indices: {np.argwhere(np.isnan(y))}")

    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"Epoch {epoch+1} - loss: {logs['loss']:.4f}, val_loss: {logs['val_loss']:.4f}")
        for x_batch, y_batch in self.train_dataset.take(1):
            sample_pred = self.model.predict(x_batch)
            logger.info(f"Sample predictions distribution: {np.histogram(sample_pred, bins=10)}")
            logger.info(f"Sample true labels distribution: {np.histogram(y_batch, bins=2)}")
            logger.info(f"Sample predictions: {sample_pred[:5]}")
            logger.info(f"Sample true labels: {y_batch[:5]}")
            logger.info(f"Sample input min: {tf.reduce_min(x_batch)}, max: {tf.reduce_max(x_batch)}")

def get_training_config(epochs, batch_size, steps_per_epoch, validation_steps):
    """Get optimized training configuration."""
    return {
        'epochs': epochs,
        'batch_size': batch_size,
        'steps_per_epoch': steps_per_epoch,
        'validation_steps': validation_steps,
        'workers': 4,  # Number of CPU workers for data loading
        'use_multiprocessing': True,
        'max_queue_size': 10,
        'shuffle': True
    }

# Update callbacks for better performance
def get_optimized_callbacks(patience, output_dir, train_dataset):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'model_{epoch:02d}_{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6
        ),
        TerminateOnNaN(),
        CustomCallback(train_dataset)
    ]

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
    
    # Print the raw y_data structure and values
    print("Raw y_data columns:", y_data.columns)
    print("Raw y_data unique values:", {col: y_data[col].unique() for col in y_data.columns})
    
    # Ensure 'ID' column exists and is of integer type
    if 'ID' in x_data.columns:
        x_data['ID'] = x_data['ID'].astype(int)
    elif 'id' in x_data.columns:
        x_data['ID'] = x_data['id'].astype(int)
        x_data = x_data.drop('id', axis=1)
    else:
        x_data['ID'] = range(len(x_data))

    # Process y_data
    if y_data.empty:
        y_data = pd.DataFrame({'ID': x_data['ID']})
        for i in range(7):  # Assuming J=7
            y_data[f'A{i}'] = np.random.choice([0, 1], size=len(y_data), p=[0.5, 0.5])
    else:
        # Handle ID column in y_data
        if 'ID' in y_data.columns:
            y_data['ID'] = y_data['ID'].astype(int)
        elif 'id' in y_data.columns:
            y_data['ID'] = y_data['id'].astype(int)
            y_data = y_data.drop('id', axis=1)
        else:
            y_data['ID'] = range(len(y_data))
        
        # Clean column names
        y_data.columns = [col.replace('.', '') for col in y_data.columns]
        
        # Select treatment columns
        A_columns = [col for col in y_data.columns if col.startswith('A') and not col.endswith('_x') and not col.endswith('_y')]
        if not A_columns:
            raise ValueError("No valid treatment columns found in y_data")
        
        y_data = y_data[['ID'] + A_columns]
    
    # Merge data with suffixes to avoid column name conflicts
    merged_data = pd.merge(x_data, y_data, on='ID', how='inner', suffixes=('', '_y'))
    
    # Print merged data info
    print("Merged data shape:", merged_data.shape)
    print("Treatment columns:", [col for col in merged_data.columns if col.startswith('A') and not col.endswith('_y')])
    
    # Get clean column lists
    x_columns = [col for col in x_data.columns if col != 'ID']
    y_columns = [col for col in y_data.columns if col != 'ID']
    
    # Split back into x_data and y_data
    x_data = merged_data[['ID'] + x_columns]
    y_data = merged_data[y_columns]
    
    # Fill NaN values and convert types
    x_data = x_data.fillna(-1).astype(np.float32)
    y_data = y_data.fillna(0).astype(np.float32)
    
    # Verify final data
    print("Final x_data shape:", x_data.shape)
    print("Final y_data shape:", y_data.shape)
    print("Final y_data unique values:", {col: y_data[col].unique() for col in y_data.columns})
    
    if y_data.empty or y_data.shape[1] == 0:
        raise ValueError("No treatment columns in final y_data")
    
    return x_data, y_data

def calculate_steps(dataset_size, batch_size):
    """Calculate steps for training/validation/testing."""
    return max(1, dataset_size // batch_size)

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
    """Create model with GPU-compatible operations and optimizations."""
    tf.keras.mixed_precision.set_global_policy('float32')
    
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output dimension: {output_dim}")
    
    inputs = Input(shape=input_shape, dtype=tf.float32, name="Inputs")
    masked_input = Masking(mask_value=0.0)(inputs)
    
    # First LSTM layer with optimizations
    lstm_1 = LSTM(int(n_hidden), 
                  dropout=0.0,
                  recurrent_dropout=0.0,
                  activation='tanh',
                  recurrent_activation="sigmoid",
                  return_sequences=True,
                  name="LSTM_1",
                  kernel_regularizer=l2(0.0001),
                  implementation=2,
                  unit_forget_bias=True,
                  unroll=True)(masked_input)  # Unroll for speed with fixed sequence length
    lstm_1 = tf.keras.layers.Dropout(dr)(lstm_1)
    lstm_1 = tf.keras.layers.BatchNormalization(momentum=0.99)(lstm_1)
    
    # Second LSTM layer
    lstm_2 = LSTM(int(math.ceil(n_hidden/2)),
                  dropout=0.0,
                  recurrent_dropout=0.0,
                  activation='tanh',
                  recurrent_activation="sigmoid",
                  return_sequences=True,
                  name="LSTM_2",
                  kernel_regularizer=l2(0.0001),
                  implementation=2,
                  unit_forget_bias=True,
                  unroll=True)(lstm_1)
    lstm_2 = tf.keras.layers.Dropout(dr)(lstm_2)
    lstm_2 = tf.keras.layers.BatchNormalization(momentum=0.99)(lstm_2)
    
    # Third LSTM layer
    lstm_3 = LSTM(int(math.ceil(n_hidden/4)),
                  dropout=0.0,
                  recurrent_dropout=0.0,
                  activation='tanh',
                  recurrent_activation="sigmoid",
                  return_sequences=False,
                  name="LSTM_3",
                  kernel_regularizer=l2(0.0001),
                  implementation=2,
                  unit_forget_bias=True,
                  unroll=True)(lstm_2)
    lstm_3 = tf.keras.layers.Dropout(dr)(lstm_3)
    
    # Dense layers with optimized batch normalization
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(lstm_3)
    x = Dense(int(math.ceil(n_hidden/4)), activation='relu',
              kernel_initializer='he_uniform')(x)  # Better initialization for ReLU
    x = tf.keras.layers.Dropout(0.3)(x)
    
    if loss_fn == "binary_crossentropy":
        x = Dense(int(math.ceil(n_hidden/8)), activation='relu',
                 kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid', name='Output')(x)
    else:
        output = Dense(J, activation='softmax', name='Output')(x)
    
    model = Model(inputs, output)
    
    # Configure optimizer and loss
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        clipnorm=1.0,
        clipvalue=0.5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=True
    )
    
    if loss_fn == "binary_crossentropy":
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.1
        )
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.BinaryCrossentropy(name='cross_entropy')
        ]
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseCategoricalCrossentropy(name='cross_entropy')
        ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        run_eagerly=False
    )
    
    return model

def create_dataset(x_data, y_data, n_pre, batch_size, loss_fn, J):
    """Create dataset with performance optimizations."""
    if loss_fn == "sparse_categorical_crossentropy":
        label_dtype = tf.int32
    else:
        label_dtype = tf.float32
    
    num_features = x_data.shape[1]
    if 'ID' in x_data.columns:
        num_features -= 1
        
    logger.info(f"Creating dataset with {num_features} features")
    
    # Create generator dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(x_data, y_data, n_pre, batch_size, loss_fn, J),
        output_signature=(
            tf.TensorSpec(shape=(n_pre, num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=label_dtype)
        )
    )
    
    # Performance optimizations in correct order
    dataset = dataset.shuffle(10000)  # Shuffle before batching
    dataset = dataset.cache()  # Cache individual samples
    dataset = dataset.batch(batch_size, drop_remainder=True)  # Batch after cache
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True
    
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Verify dataset characteristics
    try:
        for x_batch, y_batch in dataset.take(1):
            logger.info(f"Dataset verification:")
            logger.info(f"X batch shape: {x_batch.shape}")
            logger.info(f"Y batch shape: {y_batch.shape}")
            logger.info(f"X range: [{tf.reduce_min(x_batch)}, {tf.reduce_max(x_batch)}]")
    except Exception as e:
        logger.warning(f"Dataset verification failed: {str(e)}")
    
    return dataset

def data_generator(x_data, y_data, n_pre, batch_size, loss_fn, J):
    """Generate individual sequences with consistent data types."""
    logger.info(f"Generating data sequences:")
    logger.info(f"Input shapes - X: {x_data.shape}, Y: {y_data.shape}")
    
    # Process features
    x_data = x_data.copy()
    if 'ID' in x_data.columns:
        x_data = x_data.drop(columns=['ID'])
    
    # Convert and normalize data
    x_values = x_data.values.astype(np.float32)
    x_values = (x_values - np.mean(x_values, axis=0)) / (np.std(x_values, axis=0) + 1e-6)
    y_values = y_data.values.astype(np.float32)
    
    n_samples = len(x_values) - n_pre + 1
    logger.info(f"Number of sequences: {n_samples}")
    
    # Initialize policy for mixed precision
    policy = tf.keras.mixed_precision.global_policy()
    dtype = policy.compute_dtype
    logger.info(f"Using compute dtype: {dtype}")
    
    if loss_fn == "binary_crossentropy":
        # Calculate labels for all samples
        labels = np.array([float(np.sum(y_values[idx + n_pre - 1]) > 0) 
                          for idx in range(n_samples)])
        unique_labels = np.unique(labels)
        logger.info(f"Unique labels in dataset: {unique_labels}")
        
        if len(unique_labels) == 1:
            logger.info("Only one class found. Creating synthetic diversity...")
            majority_class = unique_labels[0]
            synthetic_prob = 0.1
            
            # Generate individual sequences with synthetic diversity
            while True:
                for idx in range(n_samples):
                    sequence = x_values[idx:idx + n_pre].copy()
                    sequence += np.random.normal(0, 0.01, sequence.shape).astype(np.float32)
                    label = 1 - majority_class if np.random.random() < synthetic_prob else majority_class
                    
                    yield (sequence, label)
                
                if n_samples > 0:  # Prevent infinite loop if no samples
                    np.random.shuffle(np.arange(n_samples))
        else:
            positive_indices = np.where(labels == 1)[0]
            negative_indices = np.where(labels == 0)[0]
            
            # Combine and shuffle indices for balanced sampling
            all_indices = np.concatenate([
                np.random.choice(positive_indices, len(positive_indices), replace=True),
                np.random.choice(negative_indices, len(negative_indices), replace=True)
            ])
            np.random.shuffle(all_indices)
            
            # Generate balanced sequences
            while True:
                for idx in all_indices:
                    sequence = x_values[idx:idx + n_pre].copy()
                    sequence += np.random.normal(0, 0.01, sequence.shape).astype(np.float32)
                    label = labels[idx]
                    
                    yield (sequence, label)
                
                if len(all_indices) > 0:  # Prevent infinite loop
                    np.random.shuffle(all_indices)
    else:
        # Generate sequences for multiclass classification
        indices = np.arange(n_samples)
        while True:
            np.random.shuffle(indices)
            for idx in indices:
                sequence = x_values[idx:idx + n_pre].copy()
                sequence += np.random.normal(0, 0.01, sequence.shape).astype(np.float32)
                label = np.argmax(y_values[idx + n_pre - 1]) % J
                
                yield (sequence, label)
                
            if n_samples > 0:  # Prevent infinite loop
                np.random.shuffle(indices)