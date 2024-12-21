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
        'save_model': False,
        'save_graph': False,  # Disable graph saving
        'log_weights': False,  # Disable weight logging
        'log_gradients': False,
        'training_data': None,
        'validation_data': None,
        'validation_steps': validation_steps if validation_steps else None,
        'global_step_transform': None,
        'log_graph': False,  # Explicitly disable graph logging
        'input_type': None,  # Don't try to infer input type
        'output_type': None,  # Don't try to infer output type
        'compute_flops': False,
        'batch_size': None,
        'log_evaluation': False,
        'log_batch_frequency': None  # Disable batch logging
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

# In utils.py - Replace the CustomCallback class:

class CustomCallback(tf.keras.callbacks.Callback):
    """Fixed implementation of CustomCallback"""
    
    def __init__(self, train_dataset):
        super().__init__()  # Properly initialize parent class
        self._train_dataset = train_dataset
        self._start_time = time.time()
        self._epoch_start_time = None
        self._current_model = None  # Use a different name to avoid conflicts
    
    def set_model(self, model):
        """Properly handle model setting"""
        super().set_model(model)
        self._current_model = model
               
    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}
                
        epoch_time = time.time() - self._epoch_start_time
            
        if self._current_model is None:
            logger.warning("Model not set in CustomCallback")
            return
                
        # Log metrics to WandB with safer handling
        metrics_dict = {
            'epoch': epoch,
            'epoch_time': epoch_time,
        }
        
        # Safely get learning rate
        try:
            if hasattr(self._current_model.optimizer, 'learning_rate'):
                lr = self._current_model.optimizer.learning_rate
                if hasattr(lr, 'numpy'):
                    metrics_dict['learning_rate'] = float(lr.numpy())
                elif callable(lr):
                    metrics_dict['learning_rate'] = float(lr(epoch))
        except:
            logger.warning("Could not log learning rate")
        
        # Add other logs safely
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metrics_dict[key] = value
        
        wandb.log(metrics_dict)
            
        try:
            # Sample predictions with safer handling
            for x_batch in self._train_dataset.take(1):
                if isinstance(x_batch, tuple):
                    x_batch = x_batch[0]
                    
                # Get predictions
                sample_pred = self._current_model.predict(x_batch, verbose=0)
                
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
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            mode='max'
        ),
        
        # Best model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_accuracy',
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
            filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras'),
            keep_n=3,
            save_weights_only=False,
            monitor='val_accuracy',
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
        
        # Handle target column based on type (Y, C, or A)
        if 'target' in y_data.columns:
            # Keep the target column and ID
            y_data = y_data[['ID', 'target']]
            
            # Analyze target distribution before processing
            target_dist = y_data['target'].value_counts(dropna=True)
            logger.info(f"Raw target distribution:\n{target_dist}")
            
            # Check if target is binary (Y or C)
            is_binary = len(target_dist.unique()) <= 2
            if is_binary:
                # For Y and C models: Don't fill NAs with -1
                # Instead, fill with minority class to help address class imbalance
                minority_class = target_dist.idxmin()
                y_data['target'] = y_data['target'].fillna(minority_class)
                
                # Add noise to training data to prevent perfect predictions
                noise = np.random.normal(0, 0.1, size=len(y_data))
                y_data['target'] = y_data['target'].astype(np.float32) + noise
                # Clip values to maintain binary nature while keeping some uncertainty
                y_data['target'] = np.clip(y_data['target'], 0.0, 1.0)
                
                # Log processed distribution
                logger.info("Processed target distribution (after adding noise):")
                logger.info(f"Mean: {y_data['target'].mean():.4f}")
                logger.info(f"Std: {y_data['target'].std():.4f}")
                logger.info(f"Min: {y_data['target'].min():.4f}")
                logger.info(f"Max: {y_data['target'].max():.4f}")
            else:
                # For treatment (A) model: Keep original processing
                y_data['target'] = y_data['target'].fillna(-1).astype(np.int32)
        else:
            # Handle treatment columns if no direct target column
            treatment_cols = [col for col in y_data.columns if col.startswith('A')]
            if not treatment_cols:
                raise ValueError("Neither target nor treatment columns found")
            
            # Convert one-hot encoded treatments to target
            y_data['target'] = np.argmax(y_data[treatment_cols].values, axis=1)
            y_data = y_data[['ID', 'target']]
            y_data['target'] = y_data['target'].fillna(-1).astype(np.int32)
        
        # Process input features
        # Identify binary and continuous columns
        binary_cols = []
        cont_cols = []
        
        for col in x_data.columns:
            if col != 'ID':
                unique_vals = x_data[col].nunique()
                if unique_vals <= 2:
                    binary_cols.append(col)
                else:
                    cont_cols.append(col)
        
        logger.info(f"Binary columns: {binary_cols}")
        logger.info(f"Continuous columns: {cont_cols}")
        
        # Handle binary columns
        for col in binary_cols:
            x_data[col] = x_data[col].fillna(x_data[col].mode()[0])
        
        # Handle continuous columns
        for col in cont_cols:
            # Fill NAs with median
            x_data[col] = x_data[col].fillna(x_data[col].median())
            # Add small random noise to prevent perfect correlations
            noise = np.random.normal(0, x_data[col].std() * 0.01, size=len(x_data))
            x_data[col] = x_data[col] + noise
            # Standardize
            x_data[col] = (x_data[col] - x_data[col].mean()) / x_data[col].std()
        
        # Convert types
        x_data = x_data.astype(np.float32)
        
        # Final verification
        logger.info("\nFinal data summary:")
        logger.info(f"X-data shape: {x_data.shape}")
        logger.info(f"Y-data shape: {y_data.shape}")
        logger.info(f"X-data contains NaN: {x_data.isna().any().any()}")
        logger.info(f"Y-data contains NaN: {y_data.isna().any().any()}")
        
        return x_data, y_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Input file exists: {os.path.exists(input_file)}")
        logger.error(f"Output file exists: {os.path.exists(output_file)}")
        logger.error(traceback.format_exc())
        raise
        
def configure_gpu(policy=None):
    try:
        tf.keras.backend.clear_session()
        gc.collect()

        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("No GPUs available, using CPU")
            return False

        # Use more memory on each GPU
        for gpu in gpus:
            try:
                # Use 10GB instead of 8GB
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=10 * 1024)]
                )
                logger.info(f"Successfully configured GPU: {gpu.name}")
            except RuntimeError as e:
                logger.error(f"Error configuring GPU {gpu.name}: {e}")
                continue

        # Optimize for older GPUs
        tf.config.optimizer.set_experimental_options({
            'layout_optimizer': True,
            'constant_folding': True,
            'shape_optimization': True,
            'remapping': True,
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': True,
        })

        return True

    except Exception as e:
        logger.error(f"GPU configuration failed: {e}")
        return False

# In utils.py - Replace the get_strategy function:

def get_strategy():
    """Get appropriate distribution strategy with improved GPU/CPU handling."""
    try:
        # Configure CPU threads for better performance
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.info("No GPUs available, using CPU optimization strategy")
            return tf.distribute.OneDeviceStrategy("/cpu:0")

        # Try to configure GPU if available
        try:
            for gpu in gpus:
                try:
                    # Allow memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    logger.warning(f"Could not set memory growth for GPU {gpu}: {e}")
                
                try:
                    # Set memory limit
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=8 * 1024)]
                    )
                except Exception as e:
                    logger.warning(f"Could not set memory limit for GPU {gpu}: {e}")
            
            # Get logical devices after configuration
            logical_gpus = tf.config.list_logical_devices('GPU')
            
            if len(logical_gpus) > 0:
                logger.info(f"Successfully configured {len(logical_gpus)} GPU(s)")
                if len(logical_gpus) > 1:
                    strategy = tf.distribute.MirroredStrategy()
                    logger.info("Using MirroredStrategy for multiple GPUs")
                else:
                    strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
                    logger.info("Using OneDeviceStrategy with single GPU")
                
                # Test GPU strategy
                with strategy.scope():
                    test = tf.random.uniform((100, 100))
                    result = tf.matmul(test, test)
                    del result  # Clean up test tensors
                
                return strategy
            
        except Exception as e:
            logger.warning(f"GPU configuration failed: {e}")
        
        # Fall back to CPU if GPU configuration fails
        logger.info("Falling back to CPU strategy")
        return tf.distribute.OneDeviceStrategy("/cpu:0")

    except Exception as e:
        logger.warning(f"Strategy creation failed: {e}. Using default CPU strategy")
        return tf.distribute.OneDeviceStrategy("/cpu:0")

def configure_device():
    """Configure device settings for optimal performance."""
    try:
        # Clear any existing sessions
        tf.keras.backend.clear_session()
        gc.collect()

        # Try GPU configuration first
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info("Found GPU(s), attempting to configure...")
            return configure_gpu(None)
        
        # CPU optimizations if no GPU
        logger.info("Configuring CPU optimizations...")
        
        # Enable CPU optimizations
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        os.environ['TF_CPU_DETERMINED'] = '1'
        
        # Configure thread settings
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        
        # Set memory growth if supported
        try:
            physical_devices = tf.config.list_physical_devices()
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    pass
        except:
            pass
        
        return True

    except Exception as e:
        logger.warning(f"Device configuration failed: {e}")
        return False

def create_dataset(x_data, y_data, n_pre, batch_size, loss_fn, J, is_training=False, is_censoring=False):
    """Create dataset with proper sequence handling for CPU TensorFlow."""
    logger.info(f"Creating dataset with parameters:")
    logger.info(f"n_pre: {n_pre}, batch_size: {batch_size}, loss_fn: {loss_fn}, J: {J}")
    
    try:
        # Remove ID if present
        if 'ID' in x_data.columns:
            x_data = x_data.drop('ID', axis=1)
        
        # Separate binary and continuous features
        binary_cols = []
        cont_cols = []
        
        for col in x_data.columns:
            unique_vals = x_data[col].nunique()
            if unique_vals <= 2:
                binary_cols.append(col)
            else:
                cont_cols.append(col)
                
        logger.info(f"Binary features: {len(binary_cols)}")
        logger.info(f"Continuous features: {len(cont_cols)}")
        
        # Process features separately
        x_values = x_data.values.astype(np.float32)
        
        # Only standardize continuous columns
        if cont_cols:
            cont_indices = [x_data.columns.get_loc(col) for col in cont_cols]
            mean = np.zeros(x_values.shape[1])
            std = np.ones(x_values.shape[1])
            
            mean[cont_indices] = np.nanmedian(x_values[:, cont_indices], axis=0)
            std[cont_indices] = np.nanstd(x_values[:, cont_indices], axis=0)
            std[std < 1e-6] = 1.0
        else:
            mean = np.zeros(x_values.shape[1])
            std = np.ones(x_values.shape[1])
        
        # Prepare targets based on model type
        if loss_fn == "sparse_categorical_crossentropy":
            # Treatment model (A) - keep existing logic
            if 'target' in y_data.columns:
                y_values = y_data['target'].values.astype(np.int32)
            else:
                if 'A' in y_data.columns:
                    y_values = y_data['A'].values.astype(np.int32)
                else:
                    treatment_cols = [col for col in y_data.columns if col.startswith('A')]
                    if treatment_cols:
                        y_values = np.argmax(y_data[treatment_cols].values, axis=1).astype(np.int32)
                    else:
                        raise ValueError("No target or treatment columns found")
            y_values = np.clip(y_values, 0, J-1)
        else:
            # Binary case (Y, C, or binary A models)
            if 'target' in y_data.columns:
                y_raw = y_data['target'].values.astype(np.float32)
                
                # Different handling for binary treatment vs Y/C models
                if not is_censoring and J > 1:  # Binary treatment case
                    # Convert categorical targets to one-hot encoded format
                    y_values = np.zeros((len(y_raw), J), dtype=np.float32)
                    for i in range(len(y_raw)):
                        if y_raw[i] >= 0:  # Skip -1s 
                            class_idx = min(int(y_raw[i]), J-1)
                            y_values[i, class_idx] = 1.0

                elif not is_censoring:  # Y model
                    # Convert -1s to 0s and add noise
                    y_raw = np.where(y_raw == -1, 0, y_raw)
                    pos_ratio = np.mean(y_raw > 0.5)
                    logger.info(f"Positive class ratio: {pos_ratio:.4f}")
                    
                    if is_training:
                        noise = np.random.normal(0, 0.15, size=len(y_raw))
                        y_raw = y_raw + noise
                        y_raw = np.where(y_raw > 0.5,
                                       np.clip(y_raw, 0.6, 0.95),
                                       np.clip(y_raw, 0.05, 0.4))
                    else:
                        noise = np.random.normal(0, 0.05, size=len(y_raw))
                        y_raw = y_raw + noise
                        
                    y_values = np.clip(y_raw, 0, 1)
                
                else:  # C model
                    pos_ratio = np.mean(y_raw > 0.5)
                    logger.info(f"Positive class ratio: {pos_ratio:.4f}")
                    
                    if is_training:
                        noise = np.random.normal(0, 0.1, size=len(y_raw))
                        y_raw = y_raw + noise
                        y_raw = np.where(y_raw > 0.5,
                                       np.clip(y_raw, 0.7, 0.9),
                                       np.clip(y_raw, 0.1, 0.3))
                    else:
                        noise = np.random.normal(0, 0.05, size=len(y_raw))
                        y_raw = y_raw + noise
                    
                    y_values = np.clip(y_raw, 0, 1)
            else:
                # Binary treatment model (A)
                treatment_cols = [f'A{i}' for i in range(J)]
                if all(col in y_data.columns for col in treatment_cols):
                    y_values = y_data[treatment_cols].values.astype(np.float32)
                else:
                    raise ValueError("No suitable treatment columns found")
        
        # Create sequences with improved noise handling
        num_samples = len(x_values) - n_pre + 1
        x_sequences = []
        y_sequences = []
        
        for i in range(num_samples):
            # Handle features
            x_seq = x_values[i:i + n_pre].copy()
            
            # Only standardize continuous features
            if cont_cols:
                x_seq[:, cont_indices] = np.clip(
                    (x_seq[:, cont_indices] - mean[cont_indices]) / std[cont_indices],
                    -10, 10
                )
            
            # Add noise during training
            if is_training:
                # Add more noise to continuous features
                if cont_cols:
                    noise = np.zeros_like(x_seq)
                    noise[:, cont_indices] = np.random.normal(0, 0.02, size=(n_pre, len(cont_indices)))
                    noise[:, cont_indices] = np.clip(noise[:, cont_indices], -0.05, 0.05)
                    x_seq += noise
            
            y_seq = y_values[i + n_pre - 1].copy()
            
            x_sequences.append(x_seq)
            y_sequences.append(y_seq)
        
        x_sequences = np.array(x_sequences, dtype=np.float32)
        y_sequences = np.array(y_sequences, dtype=np.float32)
        
        # Log sequence statistics
        logger.info(f"\nSequence statistics:")
        logger.info(f"X shape: {x_sequences.shape}")
        logger.info(f"Y shape: {y_sequences.shape}")
        logger.info(f"X range: [{np.min(x_sequences)}, {np.max(x_sequences)}]")
        logger.info(f"Y mean: {np.mean(y_sequences)}")
        logger.info(f"Y std: {np.std(y_sequences)}")
        
        # Create TensorFlow datasets
        dataset = tf.data.Dataset.from_tensor_slices((x_sequences, y_sequences))
        
        if is_training:
            # Improved training data handling
            dataset = dataset.shuffle(buffer_size=min(10000, len(x_sequences)), 
                                   reshuffle_each_iteration=True)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.repeat()
        else:
            dataset = dataset.batch(batch_size)
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Calculate steps
        steps = len(x_sequences) // batch_size
        if not is_training and len(x_sequences) % batch_size != 0:
            steps += 1
            
        return dataset, steps
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        logger.error(traceback.format_exc())
        raise
        
def create_model(input_shape, output_dim, lr, dr, n_hidden, hidden_activation, 
                out_activation, loss_fn, J, epochs, steps_per_epoch, y_data=None, strategy=None, is_censoring=False):
    """Create model with proper handling of binary and categorical cases.
    
    Args:
        input_shape: Shape of input tensors
        output_dim: Number of output dimensions
        lr: Learning rate
        dr: Dropout rate
        n_hidden: Number of hidden units
        hidden_activation: Activation function for hidden layers
        out_activation: Activation function for output layer
        loss_fn: Loss function (binary_crossentropy or sparse_categorical_crossentropy)
        J: Number of treatment categories
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        y_data: Optional target data for reference
        strategy: Optional distribution strategy
    """
    if strategy is None:
        strategy = get_strategy()
    
    with strategy.scope():
        # Input layer with fixed name
        inputs = Input(shape=input_shape, dtype=tf.float32, name="input_1")
        
        # Initial masking and normalization
        x = tf.keras.layers.Masking(mask_value=-1.0, name="masking_layer")(inputs)
        x = tf.keras.layers.LayerNormalization(name="norm_0")(x)
        
        # Common configurations for all LSTM layers
        lstm_config = {
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'kernel_initializer': 'glorot_uniform',
            'recurrent_initializer': 'orthogonal',
            'kernel_regularizer': l2(0.001),
            'recurrent_regularizer': l2(0.001),
            'bias_regularizer': l2(0.001),
            'kernel_constraint': None,  # Remove MaxNorm constraint
            'recurrent_constraint': None,  # Remove MaxNorm constraint
            'dropout': 0.0,
            'recurrent_dropout': 0.0,
            'dtype': tf.float32
        }
        
        # LSTM layers with gradient clipping
        x = tf.keras.layers.LSTM(
            units=n_hidden * 2,
            return_sequences=True,
            name="lstm_1",
            **lstm_config
        )(x)
        x = tf.keras.layers.LayerNormalization(name="norm_1")(x)
        x = tf.keras.layers.Dropout(dr, name="drop_1")(x)
        
        x = tf.keras.layers.LSTM(
            units=n_hidden,
            return_sequences=True,
            name="lstm_2",
            **lstm_config
        )(x)
        x = tf.keras.layers.LayerNormalization(name="norm_2")(x)
        x = tf.keras.layers.Dropout(dr, name="drop_2")(x)
        
        x = tf.keras.layers.LSTM(
            units=max(32, n_hidden // 2),
            return_sequences=False,
            name="lstm_3",
            **lstm_config
        )(x)
        x = tf.keras.layers.LayerNormalization(name="norm_3")(x)
        x = tf.keras.layers.Dropout(dr, name="drop_3")(x)
        
        # Dense layer with L2 regularization
        x = tf.keras.layers.Dense(
            units=n_hidden,
            activation='relu',
            kernel_regularizer=l2(0.001),
            name="dense_1"
        )(x)
        x = tf.keras.layers.LayerNormalization(name="norm_4")(x)
        x = tf.keras.layers.Dropout(dr, name="drop_4")(x)
        
        # Output layer configuration based on loss function
        if loss_fn == "binary_crossentropy":
            # Binary case (Y or C model)
            final_activation = 'sigmoid'
            output_units = 1 if is_censoring else J
            
            if is_censoring:
                # Special handling for compliance model with class imbalance
                pos_ratio = tf.constant(0.02, dtype=tf.float32)  # Based on observed data

                outputs = tf.keras.layers.Dense(
                    units=output_units,
                    activation=final_activation,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(0.001),
                    bias_initializer=tf.keras.initializers.Constant(np.log(pos_ratio/(1-pos_ratio))),
                    name="output_dense"
                )(x)

                # Focal loss for imbalanced case
                def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
                    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
                    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
                    modulating_factor = tf.pow(1.0 - p_t, gamma)
                    return tf.reduce_mean(alpha_factor * modulating_factor * bce)
                
                loss = focal_loss
                metrics = [
                    tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.3),
                    tf.keras.metrics.AUC(name='auc', curve='PR', from_logits=False),
                    tf.keras.metrics.Precision(name='precision', thresholds=0.3),
                    tf.keras.metrics.Recall(name='recall', thresholds=0.3)
                ]
                class_weight = {0: 1.0, 1: 50.0}  # Strong weight for minority class
            else:
                outputs = tf.keras.layers.Dense(
                    units=output_units,
                    activation=final_activation,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(0.001),
                    name="output_dense"
                )(x)

                # Regular binary case (Y model)
                loss = tf.keras.losses.BinaryCrossentropy(
                    from_logits=False,
                    label_smoothing=0.01
                )
                metrics = [
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
                class_weight = None
                
        else:
            # Categorical case (A model)
            final_activation = 'softmax'
            output_units = J
            
            x = tf.keras.layers.Dense(n_hidden//2, activation='relu')(x)

            outputs = tf.keras.layers.Dense(
                units=output_units,
                activation=final_activation,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(0.001),
                name="output_dense"
            )(x)
            
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False
            )
            
            # Appropriate metrics for categorical case
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.SparseCategoricalCrossentropy(name='cross_entropy')
            ]
            class_weight = None
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='lstm_model')
        
        # Learning rate schedule with warm-up
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=steps_per_epoch * 2,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )
        
        # Optimizer with gradient clipping
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        
        # Compile model with appropriate loss and metrics
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            weighted_metrics=['accuracy'] if class_weight is not None else None,
            jit_compile=True
        )
        
        return model

# Helper function to check if targets are binary
def is_binary_target(y_data):
    """Check if the target data is binary (0/1) or categorical."""
    if 'target' in y_data.columns:
        unique_values = np.unique(y_data['target'])
        return set(unique_values).issubset({0, 1})
    else:
        # Check treatment columns
        treatment_cols = [col for col in y_data.columns if col.startswith('A')]
        if not treatment_cols:
            return False
        treatment_values = y_data[treatment_cols].values
        return np.array_equal(treatment_values, treatment_values.astype(bool).astype(float))