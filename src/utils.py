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

class CalibrationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions for a batch
        x_batch = next(iter(self._train_dataset))[0]
        preds = self.model.predict(x_batch)
        
        # Calculate calibration metrics
        bins = np.linspace(0, 1, 11)
        binned_preds = np.digitize(preds, bins) - 1
        calibration = np.zeros(10)
        for i in range(10):
            mask = binned_preds == i
            if np.any(mask):
                calibration[i] = np.mean(preds[mask])
        
        wandb.log({
            'calibration_histogram': wandb.Histogram(calibration),
            'mean_prediction': np.mean(preds),
            'prediction_std': np.std(preds)
        }, commit=False)

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
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            mode='min'
        ),
        
        # Best model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),

        # Regular checkpoints (keep last 3)
        CustomModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras'),
            keep_n=3,
            save_weights_only=False,
            monitor='val_loss',
            mode='min'
        ),
        
        # Terminate on NaN
        tf.keras.callbacks.TerminateOnNaN(),
        
        # Custom callback for monitoring
        CustomCallback(train_dataset)
    ]
    
    return callbacks

def is_count_sequence(values):
    """Helper function to identify count sequences"""
    try:
        # Convert values to numeric
        nums = [float(x.strip()) for x in values if x.strip() and x.strip() != 'NA']
        if not nums:
            return False
            
        # Check if all values are non-negative integers
        is_int = all(float(n).is_integer() for n in nums)
        is_nonneg = all(n >= 0 for n in nums)
        
        # Look for values > 1 to distinguish from binary
        has_larger = any(n > 1 for n in nums)
        
        return is_int and is_nonneg and has_larger
    except:
        return False

def is_binary_sequence(values):
    """Helper function to identify binary sequences"""
    try:
        # Convert values to numeric, skipping NA/empty
        nums = [float(x.strip()) for x in values if x.strip() and x.strip() != 'NA']
        if not nums:
            return False
            
        # Check if only contains -1, 0, 1
        unique_vals = set(nums)
        return unique_vals.issubset({-1, 0, 1})
    except:
        return False

def load_data_from_csv(input_file, output_file):
    """
    Load and preprocess input and output data from CSV files with improved column type detection.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        
    Returns:
        tuple: Processed input and output data as pandas DataFrames
    """
    try:
        # Load input data
        x_data = pd.read_csv(input_file)

        # Drop sequence column if it exists
        if 'sequence' in x_data.columns:
            x_data = x_data.drop('sequence', axis=1)
            
        logger.info(f"Successfully loaded input data from {input_file}")
        logger.info(f"Input data shape: {x_data.shape}")
        
        # Load output data
        y_data = pd.read_csv(output_file)
        logger.info(f"Successfully loaded output data from {output_file}")
        logger.info(f"Output data shape: {y_data.shape}")
        
        # Initialize column type lists
        binary_cols = []
        cont_cols = []

        # Function to determine if a column is binary
        def is_binary_column(series):
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 2:
                # Check if values are 0/1 or boolean
                vals = set(unique_vals)
                return vals.issubset({0, 1, True, False, "0", "1"})
            return False

        # Function to determine if a column is continuous
        def is_continuous_column(col_name, series):
            try:
                # Convert to numeric, handle errors
                numeric_series = pd.to_numeric(series, errors='coerce')
                unique_vals = numeric_series.dropna().unique()
                
                # If too few unique values, probably not continuous
                if len(unique_vals) < 3:
                    return False
                    
                # Check if values have decimals
                has_decimals = any(float(x) % 1 != 0 for x in unique_vals if not pd.isna(x))
                
                # If has decimals or many unique values, likely continuous
                return has_decimals or len(unique_vals) > 10
            except:
                return False

        # First pass: identify sequences and check their content
        for col in x_data.columns:
            if col != 'ID' and isinstance(x_data[col].iloc[0], str) and ',' in str(x_data[col].iloc[0]):
                try:
                    # Get more rows for better detection (increase from 5 to 20)
                    sample_rows = [row for row in x_data[col].iloc[:20] if isinstance(row, str)]
                    if not sample_rows:
                        continue
                        
                    # Process ALL values in the sampled rows
                    all_values = []
                    for row in sample_rows:
                        values = [float(x.strip()) for x in row.split(',') 
                                 if x.strip() and x.strip() != 'NA']
                        all_values.extend(values)
                        
                    # Check unique values and their frequency
                    unique_vals = set(all_values)
                    
                    # Special handling for L1 which we know is count data
                    if col == 'L1':
                        cont_cols.append(col)
                        logger.info(f"Column {col} forced to count sequence")
                    # More thorough sequence type detection
                    elif len(unique_vals) > 2 and max(unique_vals) > 1:
                        cont_cols.append(col)
                        logger.info(f"Column {col} identified as count/continuous sequence")
                    elif unique_vals.issubset({0, 1}):
                        binary_cols.append(col)
                        logger.info(f"Column {col} identified as binary sequence")
                    else:
                        cont_cols.append(col)
                        logger.info(f"Column {col} defaulted to continuous sequence")
                        
                except (ValueError, AttributeError):
                    cont_cols.append(col)
                    logger.info(f"Column {col} defaulted to continuous (invalid sequence)")

        # Second pass: handle non-sequence columns
        for col in x_data.columns:
            if col != 'ID' and col not in binary_cols and col not in cont_cols:
                if is_binary_column(x_data[col]):
                    binary_cols.append(col)
                    logger.info(f"Column {col} identified as binary")
                elif is_continuous_column(col, x_data[col]):
                    cont_cols.append(col)
                    logger.info(f"Column {col} identified as continuous")
                else:
                    cont_cols.append(col)
                    logger.info(f"Column {col} defaulted to continuous")

        logger.info(f"Binary columns: {binary_cols}")
        logger.info(f"Continuous columns: {cont_cols}")
        
        # Handle binary columns
        for col in binary_cols:
            try:
                if isinstance(x_data[col].iloc[0], str) and ',' in str(x_data[col].iloc[0]):
                    # Process binary sequences
                    def process_binary_seq(seq_str):
                        try:
                            values = [float(x.strip()) for x in str(seq_str).split(',') if x.strip() != 'NA']
                            if not values:
                                return 0
                            # Take the mode of the sequence
                            return max(set(values), key=values.count)
                        except (ValueError, AttributeError):
                            return 0

                    x_data[col] = x_data[col].apply(process_binary_seq).fillna(0)
                else:
                    x_data[col] = pd.to_numeric(x_data[col], errors='coerce')
                    mode_val = x_data[col].mode().iloc[0] if not x_data[col].empty else 0
                    x_data[col] = x_data[col].fillna(mode_val)
                
                x_data[col] = (x_data[col] > 0).astype(float)
            except Exception as e:
                logger.error(f"Error processing binary column {col}: {str(e)}")
                x_data[col] = 0.0
        
        # Handle continuous columns
        for col in cont_cols:
            try:
                x_data[col] = pd.to_numeric(x_data[col], errors='coerce')
                med = x_data[col].median()
                if pd.isna(med):
                    med = 0.0
                x_data[col] = x_data[col].fillna(med)
                
                if not x_data[col].isna().all():
                    std = x_data[col].std()
                    if pd.isna(std) or std == 0:
                        std = 1.0
                
                    # Standardize
                    mean = x_data[col].mean()
                    std = x_data[col].std() or 1.0
                    x_data[col] = (x_data[col] - mean) / std
            except Exception as e:
                logger.error(f"Error processing continuous column {col}: {str(e)}")
                x_data[col] = 0.0
        
        # Convert to float32
        try:
            x_data = x_data.astype(np.float32)
        except Exception as e:
            logger.error(f"Error converting to float32: {str(e)}")
            # Try converting column by column
            for col in x_data.columns:
                try:
                    x_data[col] = x_data[col].astype(np.float32)
                except:
                    x_data[col] = 0.0
        
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

    logger.info("Input data shapes:")
    logger.info(f"x_data shape: {x_data.shape}")
    logger.info(f"y_data shape: {y_data.shape}")
    logger.info(f"y_data columns: {y_data.columns}")
    
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

        # Target conversion to ensure int32 for sparse_categorical_crossentropy
        if 'target' in y_data.columns:
            if loss_fn == "sparse_categorical_crossentropy":
                # Convert to int32 and stay int32
                y_data['target'] = pd.to_numeric(y_data['target'], downcast='integer')
                y_data['target'] = y_data['target'].astype('int32')

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
            # Treatment model (A) 
            if 'target' in y_data.columns:
                # Ensure consistent int32 type and force it to stay int32
                y_values = tf.cast(y_data['target'].values, tf.int32).numpy()
                # Simply ensure valid class range
                y_values = np.clip(y_values, 0, J-1)
            else:
                if 'A' in y_data.columns:
                    # Force int32 type
                    y_values = tf.cast(y_data['A'].values, tf.int32).numpy()
                else:
                    treatment_cols = [f'A{i}' for i in range(J)]
                    if all(col in y_data.columns for col in treatment_cols):
                        # Ensure consistent int32 type for argmax result
                        y_values = tf.cast(
                            np.argmax(y_data[treatment_cols].values, axis=1),
                            tf.int32
                        ).numpy()
                    else:
                        raise ValueError("No suitable treatment columns found")
            
            if not is_training:
                # For validation/test, keep exact values
                y_values = np.clip(y_values, 0, J-1)
            
            # Log class distribution
            unique, counts = np.unique(y_values, return_counts=True)
            logger.info("Treatment class distribution:")
            for val, count in zip(unique, counts):
                logger.info(f"Class {val}: {count} ({count/len(y_values)*100:.2f}%)")
        else:
            # Binary case (Y, C, or binary A models)
            if 'target' in y_data.columns:
                y_raw = y_data['target'].values.astype(np.float32)
                
                if not is_censoring and J > 1:  # Multi-class treatment case
                    # Create one-hot encoded matrix
                    y_values = np.zeros((len(y_raw), J), dtype=np.float32)
                    valid_mask = y_raw >= 0  # Identify valid entries
                    
                    # Process valid entries using vectorized operations
                    valid_indices = np.where(valid_mask)[0]
                    class_indices = np.clip(y_raw[valid_mask].astype(int), 0, J-1)
                    y_values[valid_indices, class_indices] = 1.0  # Fixed: Properly set one-hot values
                    
                    if is_training:                        
                        # Normalize each row to sum to 1
                        row_sums = np.sum(y_values, axis=1)
                        for i in range(len(y_values)):
                            if row_sums[i] > 0:
                                y_values[i] = y_values[i] / row_sums[i]
                            else:
                                y_values[i] = np.ones(J) / J
                    
                elif not is_censoring:  # Y model
                    # For Y model, keep values as 0/1 and reshape
                    y_raw = np.where(y_raw < 0, 0, y_raw)  # Convert negative values to 0
                    y_raw = np.where(y_raw > 1, 1, y_raw)  # Ensure binary values
                    pos_ratio = np.mean(y_raw > 0.5)
                    logger.info(f"Positive class ratio: {pos_ratio:.4f}")
                    
                    # Verify distribution
                    logger.info(f"Y values unique: {np.unique(y_raw, return_counts=True)}")
                    
                    # Reshape to column vector
                    y_values = y_raw.reshape(-1, 1)
                    
                    logger.info(f"Y values shape after reshape: {y_values.shape}")
                    
                else:  # C model
                    y_raw = np.where(y_raw < 0, 0, y_raw)
                    y_values = y_raw.reshape(-1, 1)
                    
            else:
                # Handle one-hot encoded treatment inputs
                treatment_cols = [f'A{i}' for i in range(J)]
                if all(col in y_data.columns for col in treatment_cols):
                    y_values = y_data[treatment_cols].values.astype(np.float32)
                    
                    if is_training:
                        # Ensure valid probability distribution
                        row_sums = y_values.sum(axis=1, keepdims=True)
                        y_values = np.where(row_sums > 0, 
                                          y_values / row_sums,
                                          np.ones_like(y_values) / J)
                else:
                    raise ValueError("No suitable treatment columns found")

        # Log target info
        logger.info(f"Target shape: {y_values.shape}")
        if len(y_values.shape) > 1:
            logger.info(f"Target mean per class: {np.mean(y_values, axis=0)}")
            logger.info(f"Target std per class: {np.std(y_values, axis=0)}")
        else:
            logger.info(f"Target mean: {np.mean(y_values)}")
            logger.info(f"Target std: {np.std(y_values)}")
                
        # Create sequences
        num_samples = len(x_values) - n_pre + 1
        x_sequences = []
        y_sequences = []

        for i in range(num_samples):
            if i + n_pre <= len(x_values):
                # Handle features
                x_seq = x_values[i:i + n_pre].copy()
                
                # Only standardize continuous features
                if cont_cols:
                    x_seq[:, cont_indices] = np.clip(
                        (x_seq[:, cont_indices] - mean[cont_indices]) / std[cont_indices],
                        -2, 2  # Reduce clipping range to prevent extreme values
                    )
                
                # Handle binary features - keep as is without standardization
                if len(binary_cols) > 0:
                    binary_indices = [x_data.columns.get_loc(col) for col in binary_cols]
                    x_seq[:, binary_indices] = x_values[i:i + n_pre, binary_indices].copy()
                
                # Ensure consistent types
                y_seq = y_values[i + n_pre - 1].copy()
                if loss_fn == "sparse_categorical_crossentropy":
                    # Explicitly force int32 type
                    y_seq = tf.cast(y_values[i + n_pre - 1], tf.int32).numpy()
                else:
                    y_seq = y_values[i + n_pre - 1].astype(np.float32)
                
                x_sequences.append(x_seq)
                y_sequences.append(y_seq)
        
        # Convert sequences to arrays with consistent types
        x_sequences = np.array(x_sequences, dtype=np.float32)
        if loss_fn == "sparse_categorical_crossentropy":
            # Force int32 type using TensorFlow cast
            y_sequences = tf.cast(y_sequences, tf.int32).numpy()
        else:
            y_sequences = np.array(y_sequences, dtype=np.float32)

        # Calculate steps once
        num_sequences = len(x_sequences)
        steps_per_epoch = math.ceil(num_sequences / batch_size)
        if steps_per_epoch < 10 and is_training:
            logger.warning(f"Very few steps ({steps_per_epoch}). Consider reducing batch size.")

        # Create dataset once
        dataset = tf.data.Dataset.from_tensor_slices((x_sequences, y_sequences))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=min(10000, num_sequences), 
                                    reshuffle_each_iteration=True)
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat()  # Infinite repeats for training
        else:
            dataset = dataset.batch(batch_size)
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Created dataset with {num_sequences} sequences")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Steps per {'epoch' if is_training else 'validation'}: {steps_per_epoch}")
        
        return dataset, steps_per_epoch
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_focal_loss(gamma=2.0, alpha=None):
    """Get focal loss function with optional alpha balancing."""
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma) * y_true
        
        # Apply class balancing if alpha provided
        if alpha is not None:
            alpha_weight = alpha * y_true + (1 - alpha) * (1 - y_true)
            focal_weight = focal_weight * alpha_weight
            
        return tf.reduce_mean(focal_weight * ce)
    return focal_loss

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
    
    # Initialize class_weight to None
    class_weight = None

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
            'kernel_regularizer': l2(0.01),
            'recurrent_regularizer': l2(0.01),
            'bias_regularizer': l2(0.01),
            'dropout': dr,
            'recurrent_dropout': 0.0,
            'unit_forget_bias': True,  # Add this to improve gradient flow
            'dtype': tf.float32
        }
        
        # Project input to common dimension for skip connections
        x = tf.keras.layers.Dense(n_hidden, activation=None, name="projection")(x)

        # First LSTM layer
        x = tf.keras.layers.LSTM(
            units=n_hidden,  # Using same dimension
            return_sequences=True,
            name="lstm_1",
            **lstm_config
        )(x)
        x = tf.keras.layers.LayerNormalization(name="norm_1")(x)
        x = tf.keras.layers.Dropout(dr, name="drop_1")(x)
        
        # Add skip connection
        skip = x

        x = tf.keras.layers.LSTM(
            units=n_hidden, # Same dimension as previous layer
            return_sequences=True,
            name="lstm_2",
            **lstm_config
        )(x)
        x = tf.keras.layers.LayerNormalization(name="norm_2")(x)
        x = tf.keras.layers.Dropout(dr, name="drop_2")(x)
        
        # Add skip connection back
        x = tf.keras.layers.Add()([x, skip])

        # Final LSTM layer for feature extraction
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
        
        # Output layer and loss configuration
        if loss_fn == "binary_crossentropy":
            final_activation = 'sigmoid'
            output_units = 1 if is_censoring else J
            
            if is_censoring:
                # Configure censoring model with focal loss
                if 'target' in y_data.columns:
                    pos_ratio = float(np.mean(y_data['target'].values > 0.5))
                    class_weight = {
                        0: 1.0,
                        1: min(10.0, 1.0/pos_ratio)  # Cap weight at 10x to prevent instability
                    }
                else:
                    pos_ratio = 0.5
                
                pos_ratio = np.clip(pos_ratio, 0.01, 0.99)
                alpha = max(0.1, min(0.9, pos_ratio))
                
                loss = get_focal_loss(gamma=2.0, alpha=alpha)
                
                outputs = tf.keras.layers.Dense(
                    units=output_units,
                    activation=final_activation,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(0.01),
                    bias_initializer=tf.keras.initializers.Constant(
                        np.log(max(1e-5, pos_ratio)/(1.0 - min(pos_ratio, 0.99999)))
                    ),
                    name="output_dense"
                )(x)
                
                metrics = [
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.AUC(name='auc', from_logits=False),
                    tf.keras.metrics.Precision(name='precision', thresholds=0.3),
                    tf.keras.metrics.Recall(name='recall')
                ]
                
                # Set censoring class weights
                neg_weight = 1.0
                pos_weight = (1.0 - pos_ratio) / (pos_ratio + 1e-7)
                class_weight = {0: neg_weight, 1: pos_weight}
                
            else:
                # Regular binary classification (Y model)
                x = tf.keras.layers.Dense(
                    units=32,
                    activation='relu',
                    kernel_regularizer=l2(0.01),
                    name="pre_output"
                )(x)
                
                # Better initializers 
                kernel_init = tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode='fan_avg', distribution='truncated_normal'
                )

                # For binary Y model - add temperature scaling for better calibration
                outputs = tf.keras.layers.Dense(
                    units=output_units,
                    activation=None,  # No activation initially
                    kernel_initializer=kernel_init,
                    kernel_regularizer=l2(0.01),
                    bias_initializer=tf.keras.initializers.Constant(0.0),  # Start from 0 bias
                    name="logits"
                )(x)

                # Add temperature scaling layer
                temperature = 2.0  # Higher temperature = softer predictions
                outputs = outputs / temperature

                # Apply sigmoid after temperature scaling
                outputs = tf.keras.layers.Activation('sigmoid', name="output_dense")(outputs)
                
                # Adjust class weights based on observed rates with better balancing
                if not is_censoring and 'target' in y_data.columns:
                    pos_count = np.sum(y_data['target'].values > 0)
                    total = len(y_data['target'])
                    pos_ratio = pos_count / total
                    
                    # More balanced weighting scheme
                    neg_weight = 1.0
                    pos_weight = (1.0 - pos_ratio) / (pos_ratio + 1e-7)
                    pos_weight = np.clip(pos_weight, 1.0, 3.0)  # Limit maximum weight
                    
                    class_weight = {
                        0: neg_weight,
                        1: pos_weight
                    }
                    
                    alpha = pos_ratio  # Use actual class ratio
                    loss = get_focal_loss(gamma=2.0, alpha=alpha)
                
                metrics = [
                    tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                    tf.keras.metrics.AUC(name='auc', curve='PR'),  # PR curve for imbalanced data
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.BinaryCrossentropy(name='cross_entropy'),
                    tf.keras.metrics.TruePositives(name='tp'),
                    tf.keras.metrics.TrueNegatives(name='tn'),
                    tf.keras.metrics.FalsePositives(name='fp'), 
                    tf.keras.metrics.FalseNegatives(name='fn')
                ]
  
        else:
            # Categorical case (A model)
            final_activation = 'softmax'
            output_units = J
            
           # Add intermediate layer to preserve individual variations
            x = tf.keras.layers.Dense(
                units=n_hidden//2,
                activation='relu',
                kernel_regularizer=l2(0.001),
                name="pre_output"
            )(x)

            outputs = tf.keras.layers.Dense(
                units=output_units,
                activation=final_activation,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(0.001),
                use_bias=True, # Enable bias for better capacity
                name="output_dense"
            )(x)
            
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False
            )
            
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.SparseCategoricalCrossentropy(name='cross_entropy')
            ]
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='lstm_model')

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=steps_per_epoch * 3,  # Longer initial period
            t_mul=2.0,
            m_mul=0.95,  # Slower decay
            alpha=0.2  # Higher minimum LR
        )
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=0.5,  # Clip gradients more aggressively
            amsgrad=True
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