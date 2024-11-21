import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import time
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint
from tensorflow.keras import backend as K

from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotUniform

import sys
import traceback

import logging

import gc
tf.keras.backend.clear_session()
gc.collect()

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import wandb
from datetime import datetime

class FilterPatterns(logging.Filter):
    def __init__(self, patterns):
        super().__init__()
        self.patterns = patterns

    def filter(self, record):
        # Return False to filter out, True to keep
        return not any(pattern in record.getMessage() for pattern in self.patterns)

def log_metrics(history, start_time):
    metrics_to_log = {}
    
    try:
        # Handle training metrics
        if 'loss' in history.history:
            metrics_to_log.update({
                'final_train_loss': float(history.history['loss'][-1]),
                'best_train_loss': float(min(history.history['loss']))
            })
        
        # Handle validation metrics
        if 'val_loss' in history.history:
            metrics_to_log.update({
                'final_val_loss': float(history.history['val_loss'][-1]),
                'best_val_loss': float(min(history.history['val_loss'])),
                'best_epoch': int(np.argmin(history.history['val_loss']))
            })
            
        # Handle accuracy metrics
        if 'accuracy' in history.history:
            metrics_to_log.update({
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'best_train_accuracy': float(max(history.history['accuracy']))
            })
            
        if 'val_accuracy' in history.history:
            metrics_to_log.update({
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'best_val_accuracy': float(max(history.history['val_accuracy']))
            })

        metrics_to_log['training_time'] = time.time() - start_time
        
    except Exception as e:
        logger.error(f"Error collecting metrics: {str(e)}")
        logger.error(traceback.format_exc())
    
    wandb.log(metrics_to_log)

def setup_wandb(config, validation_steps=None, train_dataset=None):
    """Initialize WandB with configuration.
    
    Args:
        config (dict): WandB configuration dictionary
        validation_steps (int, optional): Number of validation steps
        train_dataset: Training dataset for batch logging
    """
    run = wandb.init(
        project="multi-ltmle",
        entity="jvpoulos",
        config=config,
        name=f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Configure WandB callback with SavedModel format
    callback_config = {
        'monitor': 'val_loss',
        'log_weights': True,
        'log_gradients': True,
        'save_model': False,
        'validation_data': None,  # Remove validation data logging
        'compute_flops': False,   # Disable FLOPS computation
        'batch_size': None,       # Don't set batch size
        'log_evaluation': False,  # Disable evaluation logging
        'log_batch_frequency': 100 if train_dataset else None
    }
    
    # Add optional configurations if provided
    if validation_steps is not None:
        callback_config['validation_steps'] = validation_steps
    
    if train_dataset is not None:
        callback_config['training_data'] = train_dataset
        callback_config['log_batch_frequency'] = 100
    
    wandb_callback = wandb.keras.WandbCallback(**callback_config)
    
    return run, wandb_callback

class CustomNanCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            if np.isnan(v):
                print(f'NaN encountered in {k} at batch {batch}')
                self.model.stop_training = True
                break

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_dataset):
        super(CustomCallback, self).__init__()
        self.train_dataset = train_dataset
        self.start_time = time.time()
        self.model = None
 
    def set_model(self, model):
        super(CustomCallback, self).set_model(model)
        self.model = model
               
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}
                
        epoch_time = time.time() - self.epoch_start_time
            
        if self.model is None:
            logger.warning("Model not set in CustomCallback")
            return
                
        # Log metrics to WandB with safer handling
        metrics_dict = {
            'epoch': epoch,
            'epoch_time': epoch_time,
        }
        
        # Safely get learning rate
        try:
            metrics_dict['learning_rate'] = float(K.eval(self.model.optimizer.learning_rate))
        except:
            logger.warning("Could not log learning rate")
        
        # Add other logs safely
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metrics_dict[key] = value
        
        wandb.log(metrics_dict)
            
        try:
            # Sample predictions with safer handling
            for x_batch in self.train_dataset.take(1):
                if isinstance(x_batch, tuple):
                    x_batch = x_batch[0]
                    
                # Get predictions
                sample_pred = self.model.predict(x_batch, verbose=0)
                
                # Remove any NaN values
                sample_pred = np.nan_to_num(sample_pred, nan=0.0)
                
                if len(sample_pred.shape) > 2:
                    sample_pred = sample_pred.reshape(-1, sample_pred.shape[-1])
                
                # Only log if we have valid predictions
                if np.all(np.isfinite(sample_pred)):
                    wandb.log({
                        'sample_predictions_mean': np.mean(sample_pred),
                        'sample_predictions_std': np.std(sample_pred),
                        'sample_predictions_hist': wandb.Histogram(sample_pred.flatten())
                    })
                break
        except Exception as e:
            logger.warning(f"Error in CustomCallback prediction logging: {str(e)}")
    
    def on_train_batch_end(self, batch, logs=None):
        if not logs:
            logs = {}
            
        # Log batch-level metrics periodically
        if batch % 100 == 0:
            wandb.log({
                'batch': batch,
                'batch_loss': logs.get('loss', 0),
                'batch_accuracy': logs.get('accuracy', 0)
            })

def get_optimized_callbacks(patience, output_dir, train_dataset):
    """Get callbacks with compatible learning rate scheduling."""
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
        """Custom callback to keep only last N checkpoints."""
        def __init__(self, filepath, keep_n=3, **kwargs):
            super().__init__(filepath, **kwargs)
            self.keep_n = keep_n
            self.checkpoint_files = []
        
        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            
            # Add current checkpoint to list
            if self.filepath is not None:
                filename = self.filepath.format(epoch=epoch + 1, **logs)
                if os.path.exists(filename):
                    self.checkpoint_files.append(filename)
            
            # Remove old checkpoints if we have more than keep_n
            while len(self.checkpoint_files) > self.keep_n:
                file_to_delete = self.checkpoint_files.pop(0)
                if os.path.exists(file_to_delete):
                    try:
                        os.remove(file_to_delete)
                    except OSError as e:
                        logger.warning(f"Error deleting old checkpoint {file_to_delete}: {e}")
    
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=patience,
            restore_best_weights=True,
            mode='max'
        ),
        
        # Best model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        
        tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
        ),

        # Regular checkpoints (keep last 3)
        CustomModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.h5'),
            keep_n=3,
            save_weights_only=False,
            monitor='val_auc',
            mode='max'
        ),
        
        # Terminate on NaN
        tf.keras.callbacks.TerminateOnNaN(),
        
        # Custom callback for monitoring
        CustomCallback(train_dataset)
    ]
    
    return callbacks

def load_data_from_csv(input_file, output_file):
    try:
        # Load input data
        x_data = pd.read_csv(input_file)
        logger.info(f"Successfully loaded input data from {input_file}")
        logger.info(f"Input data shape: {x_data.shape}")
        
        # Load output data
        y_data = pd.read_csv(output_file)
        logger.info(f"Successfully loaded output data from {output_file}")
        logger.info(f"Output data shape: {y_data.shape}")
        
        # Ensure 'ID' column exists in x_data
        if 'ID' not in x_data.columns:
            if 'id' in x_data.columns:
                x_data['ID'] = x_data['id'].astype(int)
                x_data = x_data.drop('id', axis=1)
            else:
                x_data['ID'] = range(len(x_data))
        
        # Find all A (treatment) columns in y_data
        y_cols = [col for col in y_data.columns if col.startswith('A')]
        if not y_cols:
            if 'target' in y_data.columns:
                # Convert target to one-hot encoding for treatment predictions
                treatment_values = y_data['target'].values
                y_data = pd.get_dummies(treatment_values, prefix='A')
                logger.info("Converted target to one-hot treatment encoding")
        
        # Ensure column names use correct format
        y_data.columns = [f'A{i}' if col.startswith('A') else col 
                         for i, col in enumerate(y_data.columns)]
        
        logger.info("Treatment columns after processing: %s", 
                   [col for col in y_data.columns if col.startswith('A')])
        
        # Validate treatment values
        for col in y_data.columns:
            if col.startswith('A'):
                unique_vals = y_data[col].unique()
                logger.info(f"Values in {col}: {unique_vals}")
                if len(unique_vals) < 2:
                    logger.warning(f"Column {col} has no variation")
        
        # Fill NaN values and convert types
        x_data = x_data.fillna(-1).astype(np.float32)
        y_data = y_data.fillna(0).astype(np.float32)
        
        # Validate we have valid treatment assignments
        if not any(y_data.any()):
            raise ValueError("No valid treatment assignments found in y_data")
        
        return x_data, y_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Input file exists: {os.path.exists(input_file)}")
        logger.error(f"Output file exists: {os.path.exists(output_file)}")
        logger.error(traceback.format_exc())
        raise

def configure_gpu(policy=None):
    """Configure GPU to adapt to different types and numbers of GPUs."""
    try:
        # Set correct CUDA path
        cuda_path = "/n/app/cuda/11.7-gcc-9.2.0"
        
        # Basic GPU configuration
        os.environ.update({
            'CUDA_HOME': cuda_path,
            'LD_LIBRARY_PATH': f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
            'PATH': f"{cuda_path}/bin:{os.environ.get('PATH', '')}",
            'CUDA_DEVICE_ORDER': "PCI_BUS_ID",
            'CUDA_VISIBLE_DEVICES': '0,1',
            'TF_FORCE_GPU_ALLOW_GROWTH': 'true',  # Changed to true for dynamic memory allocation
            'TF_CPP_MIN_LOG_LEVEL': '3',
            'TF_CUDNN_DETERMINISTIC': '1',
            'TF_ENABLE_ONEDNN_OPTS': '0'
        })

        # Check for libdevice
        libdevice_path = f"{cuda_path}/nvvm/libdevice/libdevice.10.bc"
        if os.path.exists(libdevice_path):
            if not os.path.exists("./libdevice.10.bc"):
                try:
                    os.symlink(libdevice_path, "./libdevice.10.bc")
                except Exception as e:
                    logger.warning(f"Could not create libdevice symlink: {e}")
        else:
            logger.warning(f"libdevice not found at {libdevice_path}")

        # Reset session and clear memory
        tf.keras.backend.clear_session()
        gc.collect()

        # Configure GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    # Enable memory growth for dynamic allocation
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    try:
                        # Get GPU memory info
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        total_memory = gpu_details.get('memory_limit', 14 * 1024 * 1024 * 1024)  # Default 14GB
                        # Set memory limit to 90% of available memory
                        memory_limit = int(0.9 * total_memory / (1024 * 1024))  # Convert to MB
                    except:
                        # Fallback to default if cannot get memory info
                        memory_limit = int(14 * 1024)  # 14GB per GPU
                    
                    # Configure memory limits
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )

                # Set visible devices
                tf.config.set_visible_devices(gpus, 'GPU')
                
                # Set precision policy
                tf.keras.mixed_precision.set_global_policy('float32')
                
                # Configure for better performance
                tf.config.optimizer.set_jit(False)
                tf.config.threading.set_intra_op_parallelism_threads(2)
                tf.config.threading.set_inter_op_parallelism_threads(2)
                
                logger.info(f"GPU configuration successful. Found {len(gpus)} GPU(s)")
                for gpu in gpus:
                    logger.info(f"Using GPU: {gpu.name}")
                
                # Return appropriate strategy based on GPU count
                if len(gpus) > 1:
                    logger.info("Using MirroredStrategy for multi-GPU training")
                    return tf.distribute.MirroredStrategy()
                else:
                    logger.info("Using OneDeviceStrategy for single GPU")
                    return tf.distribute.OneDeviceStrategy(device="/gpu:0")
                
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {str(e)}")
                return tf.distribute.OneDeviceStrategy(device="/cpu:0")
        else:
            logger.warning("No GPU found, using CPU")
            return tf.distribute.OneDeviceStrategy(device="/cpu:0")
            
    except Exception as e:
        logger.warning(f"GPU configuration failed, using CPU: {str(e)}")
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")

def get_strategy():
    """Get appropriate distribution strategy based on available GPUs."""
    if not hasattr(get_strategy, "strategy"):
        try:
            # Disable XLA
            tf.config.optimizer.set_jit(False)
            
            # Force graph execution
            tf.config.run_functions_eagerly(False)
            
            # Use default strategy with synchronous updates
            get_strategy.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            
            # Configure distribution
            options = tf.distribute.InputOptions(
                experimental_fetch_to_device=False,
                experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA
            )
            get_strategy.strategy.extended.experimental_enable_get_next_as_optional = False
            
            logger.info("Using OneDeviceStrategy with synchronous updates")
        except:
            get_strategy.strategy = tf.distribute.get_strategy()
            logger.info("Using default strategy")
    
    return get_strategy.strategy

def create_dataset(x_data, y_data, n_pre, batch_size, loss_fn, J, is_training=False):
    """Create dataset with proper sequence handling."""
    logger.info(f"Creating dataset with parameters:")
    logger.info(f"n_pre: {n_pre}, batch_size: {batch_size}, loss_fn: {loss_fn}, J: {J}")
    logger.info(f"Input shape: {x_data.shape}, Output shape: {y_data.shape}")
    logger.info(f"Number of classes: {J}")
    
    # Get ID column if present before removing
    id_values = None
    if 'ID' in x_data.columns:
        id_values = x_data['ID'].values
        x_data = x_data.drop('ID', axis=1)
    
    num_features = x_data.shape[1]
    n_samples = max(0, len(x_data) - n_pre + 1)
    
    logger.info(f"Input shape: {x_data.shape}, Output shape: {y_data.shape}")
    logger.info(f"Number of classes: {J}")
    logger.info(f"Output unique values: {[y_data[col].unique() for col in y_data.columns]}")
    
    def prepare_sequences():
        x_values = x_data.values.astype(np.float32)
        
        # Calculate standardization statistics first
        mean = np.nanmedian(x_values, axis=0)
        std = np.nanstd(x_values, axis=0)
        std[std < 1e-6] = 1.0
        
        sequences_x = []
        sequences_y = []

        # Handle target values correctly for treatment assignment
        if loss_fn == "sparse_categorical_crossentropy":
            if 'target' in y_data.columns:
                y_values = y_data['target'].values
            else:
                # For one-hot encoded targets, properly convert to class indices
                treatment_cols = [col for col in y_data.columns if col.startswith('A')]
                treatment_matrix = y_data[treatment_cols].values
                
                # Ensure we're getting a proper 2D array
                if len(treatment_matrix.shape) < 2:
                    treatment_matrix = treatment_matrix.reshape(-1, 1)
                
                # Convert one-hot to indices
                try:
                    y_values = np.argmax(treatment_matrix, axis=1)
                except Exception as e:
                    logger.error(f"Error converting one-hot to indices: {e}")
                    logger.error(f"Treatment matrix shape: {treatment_matrix.shape}")
                    logger.error(f"First few rows: {treatment_matrix[:5]}")
                    raise
                
                # Validate y_values
                if not isinstance(y_values, np.ndarray):
                    y_values = np.array(y_values)
                
                # Ensure values are within expected range
                if np.any(y_values >= J):
                    logger.warning("Some class indices are larger than J-1")
                    y_values = np.clip(y_values, 0, J-1)

            # Ensure y_values is flat for bincount
            y_flat = y_values.ravel()
            class_dist = np.bincount(y_flat.astype(np.int32), minlength=J)
            logger.info(f"Treatment class distribution before sequence creation: {class_dist}")

        if loss_fn == "binary_crossentropy":
            # Convert targets to one-hot with correct shape
            if 'A' in y_data.columns:
                # Create one-hot encoding with J classes
                y_values = np.zeros((len(y_data), J), dtype=np.float32)
                a_values = y_data['A'].values.astype(int)
                for i in range(len(y_data)):
                    if a_values[i] < J:  # Only set if within valid range
                        y_values[i, a_values[i]] = 1.0
            else:
                treatment_cols = [col for col in y_data.columns if col.startswith('A')]
                y_values = y_data[treatment_cols].values.astype(np.float32)
                if y_values.shape[1] < J:
                    y_values = np.pad(y_values, ((0,0), (0, J - y_values.shape[1])))
            
            logger.info(f"Binary classification y_values shape: {y_values.shape}")
            logger.info(f"Binary classification y_values unique: {np.unique(y_values)}")
            logger.info(f"Original y_values shape: {y_values.shape}")
            logger.info(f"Original y_values unique: {np.unique(y_values)}")
        
        for i in range(n_samples):
            try:
                # Create feature sequence
                x_seq = x_values[i:i + n_pre].copy()
                x_seq = np.clip((x_seq - mean) / std, -10, 10)
                
                if is_training:
                    noise = np.random.normal(0, 0.01, x_seq.shape)
                    noise = np.clip(noise, -0.03, 0.03)
                    x_seq += noise
                
                # Get target treatment
                if loss_fn == "sparse_categorical_crossentropy":
                    y_target = y_values[i + n_pre - 1]
                    y_seq = np.array(y_target, dtype=np.int32)
                    
                    # Skip invalid sequences
                    if y_seq < 0 or y_seq >= J:
                        logger.warning(f"Invalid target value {y_seq} at position {i}, skipping")
                        continue
                if loss_fn == "binary_crossentropy": # For binary classification, keep y_seq as is
                    y_seq = y_values[i + n_pre - 1].copy()  # Copy to avoid modifying original
                    y_seq = y_seq.astype(np.float32)
                    
                sequences_x.append(x_seq.astype(np.float32))
                sequences_y.append(y_seq)
                
                # Log progress periodically
                if i % 10000 == 0:
                    logger.info(f"Processed {i} sequences")
                
            except Exception as e:
                logger.warning(f"Error processing sequence {i}: {str(e)}")
                continue
        
        if not sequences_x:
            raise ValueError("No valid sequences created")
        
        sequences_x = np.array(sequences_x, dtype=np.float32)
        sequences_y = np.array(sequences_y, dtype=np.float32)
        
        # Final validation
        if loss_fn == "sparse_categorical_crossentropy":
            class_dist = np.bincount(sequences_y.astype(np.int32), minlength=J)
            logger.info(f"Final sequences class distribution: {class_dist}")
            
            # Verify we haven't lost treatment variation
            if len(np.unique(sequences_y)) == 1:
                raise ValueError(f"All sequences mapped to same class! Distribution: {class_dist}")
            
            # Verify valid class range
            if not np.all(np.isin(sequences_y, np.arange(J))):
                invalid_classes = set(sequences_y.flatten()) - set(range(J))
                raise ValueError(f"Found invalid class indices: {invalid_classes}")
        
        logger.info(f"Final sequences_x shape: {sequences_x.shape}")
        logger.info(f"Final sequences_y shape: {sequences_y.shape}")
        logger.info(f"Final sequences_y unique values: {np.unique(sequences_y)}")
        
        return sequences_x, sequences_y
    try:
        x_sequences, y_sequences = prepare_sequences()
        
        if loss_fn == "binary_crossentropy":
            # No need to reshape y_sequences as it's already correct shape
            logger.info(f"Final shapes - x: {x_sequences.shape}, y: {y_sequences.shape}")

        # Create dataset directly - no need to reshape y_sequences since it's already correct
        dataset = tf.data.Dataset.from_tensor_slices((x_sequences, y_sequences))
        
        # Configure dataset options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        options.deterministic = False
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        dataset = dataset.with_options(options)
        
        # Suppress auto-sharding warnings
        tf.get_logger().setLevel('ERROR')

        # Apply shuffling and batching
        if is_training:
            if loss_fn == "sparse_categorical_crossentropy":
                # Calculate class weights
                class_counts = np.bincount(y_sequences.astype(np.int32).flatten(), minlength=J)
                total_samples = len(y_sequences)
                class_weights = {j: total_samples / (J * max(count, 1)) for j, count in enumerate(class_counts)}
                
                # Create sample weights
                sample_weights = np.array([class_weights[y] for y in y_sequences])
                dataset = tf.data.Dataset.from_tensor_slices((x_sequences, y_sequences, sample_weights))
            
            dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size=min(10000, n_samples), reshuffle_each_iteration=True)
        else:
            # Calculate class weights for batch
            total_samples = len(y_sequences)
            pos_samples = np.sum(y_sequences == 1)
            pos_weight = (total_samples - pos_samples) / (pos_samples + 1e-7)
            weights = np.where(y_sequences == 1, pos_weight, 1.0)
            
            # Add sample weights
            dataset = tf.data.Dataset.from_tensor_slices(
                (x_sequences, y_sequences, weights)
            )
            
            dataset = dataset.cache()
            dataset = dataset.shuffle(
                buffer_size=min(50000, n_samples),
                reshuffle_each_iteration=True,
                seed=42  # Add fixed seed for reproducibility
            )
        
        # Batch and prefetch
        dataset = dataset.batch(
            batch_size,
            drop_remainder=is_training,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        return dataset, n_samples
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        logger.error(f"x_data shape: {x_data.shape}")
        logger.error(f"y_data shape: {y_data.shape}")
        logger.error(traceback.format_exc())
        raise
        
def create_model(input_shape, output_dim, lr, dr, n_hidden, hidden_activation, out_activation, loss_fn, J, epochs, steps_per_epoch, y_data=None, strategy=None):
    """Create model with improved architecture and metrics."""
    if strategy is None:
        strategy = get_strategy()
    
    with strategy.scope():
        inputs = Input(shape=input_shape, dtype='float32', name='input_layer')
        masked = tf.keras.layers.Masking(mask_value=-1.0)(inputs)
        
        rnn_config = {
            'activation': 'tanh',
            'kernel_initializer': GlorotUniform(seed=42),
            'kernel_regularizer': l2(0.001),
            'recurrent_regularizer': l2(0.001),
            'bias_regularizer': l2(0.001),
            'kernel_constraint': tf.keras.constraints.MaxNorm(3),
            'recurrent_constraint': tf.keras.constraints.MaxNorm(3),
            'dtype': 'float32',
            'recurrent_initializer': 'orthogonal',
            'unroll': False,
            'return_sequences': True
        }
        
        x = tf.keras.layers.SimpleRNN(n_hidden * 2, name='rnn_1', **rnn_config)(masked)
        x = tf.keras.layers.LayerNormalization(name='norm_1')(x)
        x = tf.keras.layers.Dropout(dr, name='dropout_1')(x)
        
        x = tf.keras.layers.SimpleRNN(n_hidden, name='rnn_2', **rnn_config)(x)
        x = tf.keras.layers.LayerNormalization(name='norm_2')(x)
        x = tf.keras.layers.Dropout(dr, name='dropout_2')(x)
        
        x = tf.keras.layers.SimpleRNN(
            max(32, n_hidden // 2),
            return_sequences=False,
            name='rnn_3',
            **{k: v for k, v in rnn_config.items() if k != 'return_sequences'}
        )(x)
        x = tf.keras.layers.LayerNormalization(name='norm_3')(x)
        x = tf.keras.layers.Dropout(dr, name='dropout_3')(x)
        
        x = tf.keras.layers.Dense(
            n_hidden,
            activation='relu',
            kernel_regularizer=l2(0.001),
            kernel_constraint=tf.keras.constraints.MaxNorm(3),
            name='dense_1'
        )(x)
        x = tf.keras.layers.LayerNormalization(name='norm_4')(x)
        x = tf.keras.layers.Dropout(dr, name='dropout_4')(x)
        
        if loss_fn == "sparse_categorical_crossentropy":
            outputs = tf.keras.layers.Dense(
                J,
                activation='softmax',  # Use softmax for multi-class
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(0.001),  # Add L2 regularization
                kernel_constraint=tf.keras.constraints.MaxNorm(3),
                name='output_layer'
            )(x)
            
            def custom_sparse_categorical_crossentropy(y_true, y_pred):
                """Custom loss function for sparse categorical crossentropy with proper shape handling."""
                # Ensure proper shapes and types
                y_true = tf.cast(y_true, tf.int32)  # Shape: (batch_size,)
                y_true = tf.reshape(y_true, [-1])  # Ensure flat shape
                
                # Add epsilon for numerical stability
                epsilon = tf.constant(1e-7, dtype=tf.float32)
                y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)  # Shape: (batch_size, num_classes)
                
                # Get the batch size
                batch_size = tf.cast(tf.shape(y_pred)[0], tf.int32)
                
                # Ensure y_true matches the batch size
                y_true = y_true[:batch_size]
                
                # Create a mask for valid indices
                valid_mask = tf.logical_and(
                    tf.greater_equal(y_true, 0),
                    tf.less(y_true, tf.shape(y_pred)[1])
                )
                
                # Apply mask and calculate loss
                y_true = tf.boolean_mask(y_true, valid_mask)
                y_pred = tf.boolean_mask(y_pred, valid_mask)
                
                # Calculate cross entropy using TF's built-in function
                return tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        y_true,
                        y_pred,
                        from_logits=False
                    )
                )
            
            loss = custom_sparse_categorical_crossentropy
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.SparseCategoricalCrossentropy(
                    name='cross_entropy',
                    from_logits=False
                ),
                # Add class-wise accuracy metrics
                *[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name=f'class_{i}_accuracy') 
                  for i in range(J)]
            ]
        if loss_fn == "binary_crossentropy":
            outputs = tf.keras.layers.Dense(
                J,  # Change to J outputs for multi-label binary classification
                activation='sigmoid',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(0.001),
                kernel_constraint=tf.keras.constraints.MaxNorm(3),
                name='output_layer'
            )(x)

            @tf.function
            def weighted_binary_crossentropy(y_true, y_pred):
                # Ensure consistent shapes
                y_true = tf.cast(y_true, tf.float32)  # Shape: (batch_size, J)
                y_pred = tf.cast(y_pred, tf.float32)  # Shape: (batch_size, J)
                
                # Add label smoothing
                smooth_factor = 0.1
                y_true = y_true * (1.0 - smooth_factor) + smooth_factor / 2.0
                
                # Clip predictions
                epsilon = tf.constant(1e-7, dtype=tf.float32)
                y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
                
                # Binary cross entropy per class
                bce = tf.reduce_mean(
                    -(y_true * tf.math.log(y_pred) + 
                      (1.0 - y_true) * tf.math.log(1.0 - y_pred)),
                    axis=0
                )
                
                return tf.reduce_mean(bce)

            loss = weighted_binary_crossentropy
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                tf.keras.metrics.AUC(name='auc', curve='ROC'),
                tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.BinaryCrossentropy(name='bce')
            ]
        
        model = Model(inputs, outputs, name='lstm_model')
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=lr,
                first_decay_steps=steps_per_epoch * 2,  # Slower decay
                t_mul=2.0,
                m_mul=0.97,  # Gentler decay
                alpha=0.1    # Higher minimum LR
            ),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0,   # Increased from 0.5
            amsgrad=True
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            run_eagerly=False,
            jit_compile=False
        )
        
        model.summary()
        return model