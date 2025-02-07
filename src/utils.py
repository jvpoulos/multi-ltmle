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

def get_data_filenames(is_censoring, loss_fn, outcome_cols, output_dir=None):
    """Get appropriate input/output filenames based on model type."""
    if is_censoring:
        logger.info("Using censoring model filenames")
        base = "lstm_bin_C"
        return f"{base}_input.csv", f"{base}_output.csv"
    else:
        # Only proceed to Y model after C model is done
        if (isinstance(outcome_cols, list) and any(col.startswith('Y') for col in outcome_cols)) or \
           (isinstance(outcome_cols, str) and outcome_cols.startswith('Y')):
            # First check if C model exists and has been trained
            if output_dir is not None:
                c_input = os.path.join(output_dir, "lstm_bin_C_input.csv")
                c_model = os.path.join(output_dir, "lstm_bin_C_model.keras")
                c_preds = os.path.join(output_dir, "lstm_bin_C_preds.npy")
                
                # If C model files don't exist or training not complete, train C first
                if not all(os.path.exists(f) for f in [c_input, c_model, c_preds]):
                    logger.info("C model not found or incomplete, training C model first...")
                    return "lstm_bin_C_input.csv", "lstm_bin_C_output.csv"
            
            logger.info("Using Y model filenames") 
            base = "lstm_bin_Y"
        else:
            # Treatment models
            if loss_fn == "sparse_categorical_crossentropy":
                base = "lstm_cat_A"
            else:
                base = "lstm_bin_A"
    
    logger.info(f"Using base filename: {base}")
    return f"{base}_input.csv", f"{base}_output.csv"
    
def log_metrics(history, start_time):
    metrics_to_log = {}
    
    try:
        if 'cross_entropy' in history.history:
            metrics_to_log.update({
                'final_cross_entropy': float(history.history['cross_entropy'][-1]),
                'best_cross_entropy': float(min(history.history['cross_entropy'])),
                'best_epoch': int(np.argmin(history.history['cross_entropy']))
            })
            
        if 'val_cross_entropy' in history.history:
            metrics_to_log.update({
                'final_val_cross_entropy': float(history.history['val_cross_entropy'][-1]),
                'best_val_cross_entropy': float(min(history.history['val_cross_entropy'])),
                'best_epoch': int(np.argmin(history.history['val_cross_entropy']))
            })

        if 'accuracy' in history.history:
            metrics_to_log.update({
                'final_accuracy': float(history.history['accuracy'][-1]),
                'best_accuracy': float(min(history.history['accuracy'])),
                'best_epoch': int(np.argmin(history.history['accuracy']))
            })
            
        if 'val_accuracy' in history.history:
            metrics_to_log.update({
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'best_val_accuracy': float(min(history.history['val_accuracy'])),
                'best_epoch': int(np.argmin(history.history['val_accuracy']))
            })

        if 'loss' in history.history:
            metrics_to_log.update({
                'final_loss': float(history.history['loss'][-1]),
                'best_loss': float(min(history.history['loss'])),
                'best_epoch': int(np.argmin(history.history['loss']))
            })
            
        if 'val_loss' in history.history:
            metrics_to_log.update({
                'final_val_loss': float(history.history['val_loss'][-1]),
                'best_val_loss': float(min(history.history['val_loss'])),
                'best_epoch': int(np.argmin(history.history['val_loss']))
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

        # Regular checkpoints (keep last 1)
        CustomModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras'),
            keep_n=1,
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
                
                # Remove standardization here - will be done in create_dataset
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

def create_dataset(x_data, y_data, n_pre, batch_size, loss_fn, J, is_training=False, is_censoring=False, pos_weight=None, neg_weight=None):
    """Create dataset with proper sequence handling for CPU TensorFlow."""
    logger.info(f"Creating dataset with parameters:")
    logger.info(f"n_pre: {n_pre}, batch_size: {batch_size}, loss_fn: {loss_fn}, J: {J}")
    logger.info(f"Input shape at start - features: {x_data.shape[1]}")

    # Remove ID if present
    if 'ID' in x_data.columns:
        x_data = x_data.drop('ID', axis=1)
    
    # Define base features
    a_model_features = ['V3', 'white', 'black', 'latino', 'other', 'mdd', 'bipolar', 'schiz', 'L1', 'L2', 'L3']
    x_data = x_data[a_model_features].copy()
    
    logger.info(f"Selected features: {a_model_features}")
    logger.info(f"Raw L1 stats - mean: {x_data['L1'].mean():.3f}, std: {x_data['L1'].std():.3f}")
    
    # Process features
    x_values = x_data.values.astype(np.float32)
    n_features = x_values.shape[1]
    logger.info(f"Features after processing: {n_features}")

    # Separate scaling for different feature types
    # Continuous non-count features
    cont_cols = ['V3']
    cont_indices = [x_data.columns.get_loc(col) for col in cont_cols if col in x_data.columns]
    
    if cont_indices:
        for idx in cont_indices:
            col_values = x_values[:, idx]
            # Use robust scaling with median and IQR
            q75, q25 = np.percentile(col_values, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1.0
            median = np.median(col_values)
            x_values[:, idx] = (col_values - median) / (iqr + 1e-6)

    # Special handling for L1 (count data)
    l1_idx = x_data.columns.get_loc('L1')
    l1_values = x_values[:, l1_idx]

    # Add debug logging
    logger.info(f"L1 before processing - min: {np.min(l1_values)}, max: {np.max(l1_values)}")
    logger.info(f"L1 unique values: {np.unique(l1_values)}")

    if np.any(l1_values != 0):  # Only scale if non-zero values exist
        # Log transform for count data (adding 1 to handle zeros)
        l1_transformed = np.log1p(l1_values)
        # Scale to [-1, 1] range using robust scaling
        if np.std(l1_transformed) != 0:
            # Use robust scaling with median and IQR for L1
            q75, q25 = np.percentile(l1_transformed, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1.0
            median = np.median(l1_transformed)
            x_values[:, l1_idx] = (l1_transformed - median) / (iqr + 1e-6)
        else:
            logger.warning("L1 has zero standard deviation after transformation")
    else:
        logger.warning("L1 contains all zeros")

    # Clip all features to prevent extreme values
    x_values = np.clip(x_values, -3, 3)

    # Process targets with robust handling
    if loss_fn == "sparse_categorical_crossentropy":
        if 'target' in y_data.columns:
            y_values = y_data['target'].values
        else:
            if 'A' in y_data.columns:
                y_values = y_data['A'].values
            else:
                treatment_cols = [f'A{i}' for i in range(J)]
                if all(col in y_data.columns for col in treatment_cols):
                    y_values = np.argmax(y_data[treatment_cols].values, axis=1)
                else:
                    raise ValueError("No suitable treatment columns found")
        
        y_values = np.clip(y_values, 0, J-1).astype(np.int32)
    else:  # binary_crossentropy
        if 'target' in y_data.columns:
            y_values = y_data['target'].values
        else:
            # Handle multi-class binary case
            if all(f'A{i}' in y_data.columns for i in range(J)):
                y_values = y_data[[f'A{i}' for i in range(J)]].values
            else:
                raise ValueError("No suitable target columns found")

    # Create sequences with proper dimensionality
    num_samples = len(x_values) - n_pre + 1
    
    # Create normalized time and position encodings
    time_index = (np.arange(len(x_values))[:, np.newaxis] / max(len(x_values)-1, 1)).astype(np.float32)
    pos_encoding = (np.arange(n_pre)[:, np.newaxis] / max(n_pre-1, 1)).astype(np.float32)
    
    # Add time index to features - scaled between -1 and 1
    x_values_with_time = np.concatenate([x_values, 2 * time_index - 1], axis=1)
    logger.info(f"Shape after adding time: {x_values_with_time.shape}")
    
    # Initialize sequences array with correct dimensions
    x_sequences = np.zeros((num_samples, n_pre, n_features + 2), dtype=np.float32)

    # Different initialization for y_sequences based on loss function
    if loss_fn == "sparse_categorical_crossentropy":
        # For categorical A model
        y_sequences = np.zeros(num_samples, dtype=np.int32)
    else:  # binary_crossentropy
        if len(y_values.shape) > 1:
            # For multi-class binary (binary A model)
            y_sequences = np.zeros((num_samples, y_values.shape[1]), dtype=np.float32)
        else:
            # For single binary output (C/Y model)
            y_sequences = np.zeros((num_samples, 1), dtype=np.float32)

    # Fill sequences with proper indexing
    for i in range(num_samples):
        end_idx = i + n_pre
        if end_idx <= len(x_values):
            x_sequences[i, :, :n_features + 1] = x_values_with_time[i:end_idx]
            x_sequences[i, :, -1] = 2 * pos_encoding[:, 0] - 1
            if loss_fn == "sparse_categorical_crossentropy":
                # For categorical A model
                y_sequences[i] = y_values[i]
            elif len(y_values.shape) > 1:
                # For multi-class binary (binary A model)
                y_sequences[i, :] = y_values[i]
            else:
                # For single binary output (C/Y model)
                y_sequences[i, 0] = y_values[i]

    # Final preprocessing
    x_sequences = np.nan_to_num(x_sequences, nan=0.0, posinf=1.0, neginf=-1.0)

    logger.info(f"Final features dimension: {n_features + 2}") # Original features + time + position

    # Log shapes and stats
    logger.info(f"Final sequence shapes:")
    logger.info(f"X sequences: {x_sequences.shape}")
    logger.info(f"Y sequences: {y_sequences.shape}")
    logger.info(f"X range: [{np.min(x_sequences):.3f}, {np.max(x_sequences):.3f}]")
    logger.info(f"Final feature dimension: {x_sequences.shape[2]}")
    logger.info(f"Feature stats:")
    for i in range(x_sequences.shape[2]):
        data = x_sequences[:, :, i].flatten()
        logger.info(f"  Feature {i}: mean={np.mean(data):.3f}, std={np.std(data):.3f}")
        
    # Create dataset with memory optimizations
    dataset = tf.data.Dataset.from_tensor_slices((x_sequences, y_sequences))

    # Add weights if needed
    if not is_censoring and 'target' in y_data.columns:
        sequence_weights = np.linspace(0.5, 1.0, n_pre)
        sample_weights = (y_sequences >= 0).astype(np.float32)
        sample_weights *= sequence_weights[-1]
        dataset = tf.data.Dataset.zip((
            dataset,
            tf.data.Dataset.from_tensor_slices(sample_weights)
        )).map(lambda xy, w: (xy[0], xy[1], w))
    elif is_censoring:
        dataset = dataset.map(lambda x, y: (x, y, tf.ones_like(y, dtype=tf.float32)))

    # Memory-efficient pipeline
    
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(10000, num_samples),
            reshuffle_each_iteration=True
        )

    dataset = (dataset
              .batch(batch_size, drop_remainder=is_training)
              .cache()  # Cache after batching
              .prefetch(tf.data.AUTOTUNE)
              .repeat())  # Repeat at the end

    logger.info(f"Dataset created successfully")
    logger.info(f"Features: {n_features}, With time & position: {n_features + 2}")

    return dataset, num_samples

def weighted_binary_crossentropy(pos_weight, neg_weight):
    """Custom weighted binary crossentropy that explicitly accepts class weights"""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        
        # Clip prediction values to avoid log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate binary crossentropy
        bce = -(y_true * tf.math.log(y_pred) * pos_weight +
                (1 - y_true) * tf.math.log(1 - y_pred) * neg_weight)
        
        return tf.reduce_mean(bce)
    return loss

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, name="multi_head_attention", **kwargs):
        super().__init__(name=name, **kwargs)
        
    def call(self, queries, keys, values, mask=None):
        # Multi-head attention with scaling
        matmul_qk = tf.matmul(queries, keys, transpose_b=True)
        dk = tf.cast(tf.shape(keys)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, values)
        
        return output, attention_weights

def create_model(input_shape, output_dim, lr, dr, n_hidden, hidden_activation, 
                out_activation, loss_fn, J, epochs, steps_per_epoch, y_data=None, strategy=None, is_censoring=False):
    logger.info(f"Model input shape: {input_shape}")
    if strategy is None:
        strategy = get_strategy()
    
    class_weight = None

    with strategy.scope():
        inputs = Input(shape=input_shape, dtype=tf.float32, name="input_1")
        
        # Initial masking and normalization 
        x = tf.keras.layers.Masking(mask_value=-1.0, name="masking_layer")(inputs)
        x = tf.keras.layers.LayerNormalization(name="norm_0")(x)
        
        # Add positional embedding
        pos_embedding = tf.keras.layers.Dense(
            units=n_hidden,
            name="positional_embedding"
        )(x)
        
        # Enhanced LSTM config
        lstm_config = {
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid', 
            'kernel_initializer': 'glorot_uniform',
            'recurrent_initializer': 'orthogonal',
            'kernel_regularizer': l2(0.01),
            'recurrent_regularizer': l2(0.01),
            'bias_regularizer': l2(0.01),
            'dropout': dr,
            'recurrent_dropout': 0,
            'unit_forget_bias': True,
            'dtype': tf.float32
        }

        # Multi-head self-attention layer
        def attention_layer(queries, keys, values, mask=None):
            # Multi-head attention with scaling
            matmul_qk = tf.matmul(queries, keys, transpose_b=True)
            dk = tf.cast(tf.shape(keys)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            
            if mask is not None:
                scaled_attention_logits += (mask * -1e9)
            
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, values)
            return output, attention_weights

        # First LSTM + Attention block
        lstm1 = tf.keras.layers.LSTM(
            units=n_hidden,
            return_sequences=True,
            name="lstm_1",
            **lstm_config
        )(x)
        
        # Project positional embedding to match first LSTM
        pos_emb1 = pos_embedding

        # Self-attention on LSTM outputs
        attention_layer = MultiHeadAttention(name="attention_1")
        att1, _ = attention_layer(
            queries=lstm1 + pos_emb1,
            keys=lstm1 + pos_emb1,
            values=lstm1
        )
        x = tf.keras.layers.LayerNormalization(name="norm_1")(att1)
        x = tf.keras.layers.Dropout(dr, name="drop_1")(x)
        
        # Skip connection with attention output
        skip = x

        # Second LSTM + Attention block
        lstm2 = tf.keras.layers.LSTM(
            units=n_hidden,
            return_sequences=True,
            name="lstm_2",
            **lstm_config
        )(x)
        
        # Project positional embedding to match second LSTM
        pos_emb2 = pos_embedding
        
        # Self-attention using custom layer
        attention_layer2 = MultiHeadAttention(name="attention_2")
        att2, _ = attention_layer2(
            queries=lstm2 + pos_emb2,
            keys=lstm2 + pos_emb2,
            values=lstm2
        )
        
        x = tf.keras.layers.LayerNormalization(name="norm_2")(att2)
        x = tf.keras.layers.Dropout(dr, name="drop_2")(x)
        
        # Residual connection
        x = tf.keras.layers.Add()([x, skip])

        # Final LSTM + Attention for sequence aggregation
        lstm3 = tf.keras.layers.LSTM(
            units=max(32, n_hidden // 2),
            return_sequences=True,
            name="lstm_3",
            **lstm_config
        )(x)
        
        # Project positional embedding to match third LSTM
        pos_emb3 = tf.keras.layers.Dense(
            units=max(32, n_hidden // 2),
            name="positional_embedding_3"
        )(pos_embedding)
        
        # Global attention using custom layer
        attention_layer3 = MultiHeadAttention(name="attention_3")
        att3, _ = attention_layer3(
            queries=lstm3 + pos_emb3,
            keys=lstm3 + pos_emb3,
            values=lstm3
        )
        
        # Global average pooling with attention weights
        x = tf.keras.layers.GlobalAveragePooling1D()(att3)
        x = tf.keras.layers.LayerNormalization(name="norm_3")(x)
        x = tf.keras.layers.Dropout(dr, name="drop_3")(x)

        # Dense layer with attention-aware features
        x = tf.keras.layers.Dense(
            units=n_hidden,
            activation='relu',
            kernel_regularizer=l2(0.005),
            name="dense_1"
        )(x)
        x = tf.keras.layers.LayerNormalization(name="norm_4")(x)
        x = tf.keras.layers.Dropout(dr, name="drop_4")(x)

        # Initialize outputs variable
        outputs = None
        loss = None
        metrics = None
        
        # Output layer configuration based on loss function
        if loss_fn == "binary_crossentropy":
            final_activation = 'sigmoid'
            output_units = 1 if is_censoring else J
            init_bias = 0.0
            if is_censoring:
                if 'target' in y_data.columns:
                    # For censoring model, censored (target=-1) should be mapped to 1, uncensored to 0
                    valid_mask = y_data['target'].values != -1  # Identify uncensored entries
                    n_censored = np.sum(~valid_mask)
                    n_total = len(y_data['target'].values)
                    pos_ratio = n_censored / n_total  # Ratio of censored entries
                    pos_ratio = np.clip(pos_ratio, 0.01, 0.99)
                    
                    init_bias = np.log(pos_ratio / (1.0 - pos_ratio))
                    
                    pos_weight = 1.0 / (pos_ratio + 1e-7) * 0.5 # Scale down positive weight
                    neg_weight = 1.0 / (1.0 - pos_ratio + 1e-7)
                    total = pos_weight + neg_weight  
                    pos_weight = pos_weight / total * 2
                    neg_weight = neg_weight / total * 2
                    
                    class_weight = {0: neg_weight, 1: pos_weight}
                    
                    logger.info(f"Censoring model class weights:")
                    logger.info(f"Censored (weight={pos_weight:.4f}): {n_censored}")
                    logger.info(f"Uncensored (weight={neg_weight:.4f}): {n_total - n_censored}")
                    
                    loss = weighted_binary_crossentropy(
                        pos_weight=pos_weight,
                        neg_weight=neg_weight
                    )
                    
                    outputs = tf.keras.layers.Dense(
                        units=output_units,
                        activation=final_activation,
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=l2(0.005),
                        bias_initializer=tf.keras.initializers.Constant(init_bias),
                        name="output_dense"
                    )(x)
                    
                    metrics = [
                        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.BinaryCrossentropy(name='cross_entropy')
                    ]
            else:
                if output_units == 1:
                    # Single binary output
                    init_bias = 0.0
                    if 'target' in y_data.columns:
                        pos_ratio = float(np.mean(y_data['target'].values > 0.5))
                        init_bias = np.log(pos_ratio / (1.0 - pos_ratio))
                else:
                    # Multiple binary outputs
                    init_bias = [0.0] * output_units
                    if y_data is not None:
                        onehot_cols = [f'A{i}' for i in range(output_units)]
                        if all(col in y_data.columns for col in onehot_cols):
                            init_bias = [np.log(np.mean(y_data[col]) / (1 - np.mean(y_data[col]) + 1e-7)) 
                                       for col in onehot_cols]

                outputs = tf.keras.layers.Dense(
                    units=output_units,
                    activation=final_activation,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(0.005),
                    name="output_dense"
                )(x)

                loss = tf.keras.losses.BinaryCrossentropy()
                metrics = [
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.AUC(name='auc', multi_label=True if output_units > 1 else False),
                    tf.keras.metrics.BinaryCrossentropy(name='cross_entropy')
                ]

        else:  # sparse_categorical_crossentropy
            final_activation = 'softmax'
            output_units = J
            
            outputs = tf.keras.layers.Dense(
                units=output_units,
                activation=final_activation,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(0.01),
                use_bias=True,
                name="output_dense"
            )(x)
            
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False
            )
            
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.SparseCategoricalCrossentropy(name='cross_entropy')
            ]
        
        if outputs is None:
            raise ValueError("Outputs layer was not properly created. Check loss function configuration.")

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='lstm_model')

        # Learning rate schedule with longer warmup
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=steps_per_epoch * 10,  # Longer initial decay
            t_mul=2.0,  # Double period each restart
            m_mul=0.9,  # Slightly reduce max learning rate
            alpha=0.1  # Minimum learning rate
        )
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0,
            amsgrad=True
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            weighted_metrics=['accuracy'] if class_weight is not None else None,
            jit_compile=True
        )
        
        return model