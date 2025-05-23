import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import time
import json
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

from datetime import datetime

# Import wandb conditionally
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


def get_model_filenames_test(loss_fn, output_dim, is_censoring):
    """Get appropriate filenames for model and predictions based on model type."""
    
    if is_censoring:
        model_filename = 'lstm_bin_C_model.keras'
        pred_filename = 'test_bin_C_preds.npy'  # Remove lstm_ prefix
        info_filename = 'test_bin_C_preds_info.npz'
    else:
        # Check if Y model based on dimensions and loss function
        is_y_model = (output_dim == 1 and loss_fn == "binary_crossentropy")
        if is_y_model:
            model_filename = 'lstm_bin_Y_model.keras'
            pred_filename = 'test_bin_Y_preds.npy'  # Remove lstm_ prefix
            info_filename = 'test_bin_Y_preds_info.npz'
        else:
            # Treatment model (A)
            if loss_fn == "sparse_categorical_crossentropy":
                model_filename = 'lstm_cat_A_model.keras'
                pred_filename = 'test_bin_A_preds.npy'  # Remove lstm_ prefix
                info_filename = 'test_bin_A_preds_info.npz'
            else:
                model_filename = 'lstm_bin_A_model.keras'
                pred_filename = 'test_bin_A_preds.npy'  # Remove lstm_ prefix 
                info_filename = 'test_bin_A_preds_info.npz'
    
    return model_filename, pred_filename, info_filename

def get_model_filenames(loss_fn, output_dim, is_censoring):
    """Get appropriate filenames for model and predictions based on model type."""
    
    if is_censoring:
        model_filename = 'lstm_bin_C_model.keras'
        pred_filename = 'lstm_bin_C_preds.npy'
        info_filename = 'lstm_bin_C_preds_info.npz'
    else:
        # Check if Y model based on dimensions and loss function
        is_y_model = (output_dim == 1 and loss_fn == "binary_crossentropy")
        if is_y_model:
            model_filename = 'lstm_bin_Y_model.keras'
            pred_filename = 'lstm_bin_Y_preds.npy'
            info_filename = 'lstm_bin_Y_preds_info.npz'
        else:
            # Treatment model (A)
            if loss_fn == "sparse_categorical_crossentropy":
                model_filename = 'lstm_cat_A_model.keras'
                pred_filename = 'lstm_cat_A_preds.npy'
                info_filename = 'lstm_cat_A_preds_info.npz'
            else:
                model_filename = 'lstm_bin_A_model.keras'
                pred_filename = 'lstm_bin_A_preds.npy'
                info_filename = 'lstm_bin_A_preds_info.npz'
    
    return model_filename, pred_filename, info_filename

def save_model_components(model, base_path):
    """Save model architecture and weights separately."""
    try:
        # Strip metrics from the model before saving
        original_metrics = model.metrics
        model._metrics = []  # Clear metrics
        
        # Get the custom objects dictionary for core components
        custom_objects = {
            'MultiHeadAttention': MultiHeadAttention,
            'masked_binary_crossentropy': masked_binary_crossentropy,
            'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy
        }
        
        # Save stripped model architecture
        json_config = model.to_json()
        arch_path = f"{base_path}.json"
        with open(arch_path, 'w') as f:
            f.write(json_config)
            
        # Save model metadata
        meta_path = f"{base_path}.meta.json"
        meta_info = {
            'metrics': [m.name if hasattr(m, 'name') else str(m) for m in original_metrics],
            'loss_name': model.loss.name if hasattr(model.loss, 'name') else str(model.loss)
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f)
            
        logger.info(f"Model architecture saved to: {arch_path}")
        logger.info(f"Model metadata saved to: {meta_path}")
        
        # Save the weights with proper .h5 extension
        weights_path = f"{base_path}.weights.h5"
        model.save_weights(weights_path)
        logger.info(f"Model weights saved to: {weights_path}")
        
        # Restore original metrics
        model._metrics = original_metrics
        
    except Exception as e:
        logger.error(f"Error saving model components: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def masked_binary_crossentropy(y_true, y_pred):
    """
    Custom loss function that handles masked/censored values for binary classification.
    
    Args:
        y_true: True labels with -1 indicating censoring
        y_pred: Predicted probabilities
    
    Returns:
        Mean binary cross-entropy over valid elements
    """
    # Convert inputs to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Create mask for non-censored values (True where y_true != -1)
    mask = tf.not_equal(y_true, -1)
    mask = tf.cast(mask, tf.float32)
    
    # Replace -1 values with 0 to avoid invalid BCE computation
    y_true_masked = tf.where(mask > 0, y_true, tf.zeros_like(y_true))
    
    # Compute binary cross-entropy
    bce = tf.keras.losses.binary_crossentropy(
        y_true_masked,
        y_pred,
        from_logits=False,
        axis=-1
    )
    
    # For multi-label case, average across labels
    if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
        mask_sum = tf.maximum(tf.reduce_sum(mask, axis=-1), 1.0)
        masked_bce = bce * tf.reduce_mean(mask, axis=-1)
        return tf.reduce_sum(masked_bce) / tf.cast(tf.shape(y_true)[0], tf.float32)
    else:
        # For single label case
        return tf.reduce_sum(bce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """Custom loss function for sparse categorical crossentropy with masking."""
    # Create mask for non-censored values 
    mask = tf.not_equal(y_true, -1)
    
    # Extract valid samples
    valid_y_true = tf.boolean_mask(y_true, mask)
    valid_y_pred = tf.boolean_mask(y_pred, mask)
    
    # Calculate loss on valid samples
    scce = tf.keras.losses.sparse_categorical_crossentropy(
        valid_y_true,
        valid_y_pred
    )
    
    # Average over valid samples
    n_valid = tf.reduce_sum(tf.cast(mask, tf.float32))
    return tf.reduce_sum(scce) / (n_valid + K.epsilon())


# MaskedBinaryAccuracy removed - functionality covered by MaskedAccuracy class

# MaskedAUC removed - functionality covered by standard AUC with properly applied mask

@tf.keras.utils.register_keras_serializable(package='Custom')
class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='masked_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true > 0, tf.float32)
        
        correct_predictions = tf.cast(tf.equal(y_true, y_pred), tf.float32) * mask
        self.correct.assign_add(tf.reduce_sum(correct_predictions))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_states(self):
        self.correct.assign(0.)
        self.total.assign(0.)

    def get_config(self):
        return super().get_config()

def clean_model_config(config):
    """Remove metrics and loss functions from model config to ensure clean loading."""
    if isinstance(config, dict):
        # Create a copy to avoid modifying the original during iteration
        cleaned_config = config.copy()

        # Remove metrics and loss from compile config
        if 'compile_config' in cleaned_config:
            compile_config = cleaned_config['compile_config']
            if isinstance(compile_config, dict):
                # Remove problematic fields safely
                compile_config.pop('metrics', None)
                compile_config.pop('loss', None)
                compile_config.pop('loss_weights', None)
                compile_config.pop('weighted_metrics', None)

        # Recursively clean nested dictionaries
        for key, value in cleaned_config.items():
            if isinstance(value, (dict, list)):
                cleaned_config[key] = clean_model_config(value)

        return cleaned_config
    elif isinstance(config, list):
        # Recursively clean lists
        return [clean_model_config(item) for item in config]
    else:
        # Return primitives unchanged
        return config

def load_model_components(base_path, loss_fn, is_censoring, gbound=None, ybound=None):
    """Load model with proper custom objects."""
    try:
        # Define custom objects - minimal set
        custom_objects = {
            'MultiHeadAttention': MultiHeadAttention,
            'masked_binary_crossentropy': masked_binary_crossentropy
        }

        # Load and clean model architecture
        arch_path = f"{base_path}.json"
        weights_path = f"{base_path}.weights.h5"
        
        logger.info(f"Loading model architecture from: {arch_path}")
        with open(arch_path) as f:
            model_config = json.load(f)
        
        # Clean the config
        cleaned_config = clean_model_config(model_config)
        model_json = json.dumps(cleaned_config)
        
        # Load model from cleaned config
        model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
        
        logger.info(f"Loading weights from: {weights_path}")
        model.load_weights(weights_path)
        
        # Recompile model with fresh metrics
        model.compile(
            optimizer='adam',
            loss=masked_binary_crossentropy,
            metrics=[MaskedAccuracy(name='accuracy')],
            run_eagerly=True
        )
        
        logger.info("Model loaded and compiled successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error(f"Architecture path exists: {os.path.exists(arch_path)}")
        logger.error(f"Weights path exists: {os.path.exists(weights_path)}")
        raise

def create_temporal_split(x_data, y_data, n_pre, train_frac=0.8, val_frac=0.1):
    """Create temporally-aware train/val/test splits.

    Args:
        x_data: Input features DataFrame
        y_data: Target values DataFrame
        n_pre: Sequence window size
        train_frac: Fraction of sequences for training
        val_frac: Fraction of sequences for validation
    """

    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be between 0 and 1")

    if not (0 <= val_frac < 1):
        raise ValueError("val_frac must be between 0 and 1")

    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1")
    # Calculate number of complete sequences
    num_sequences = len(x_data) - n_pre + 1
    
    # Calculate split points based on sequences
    train_sequences = int(num_sequences * train_frac)
    val_sequences = int(num_sequences * val_frac)
    
    # Add window_size-1 to keep complete sequences
    train_idx = train_sequences + n_pre - 1
    val_idx = train_idx + val_sequences
    
    # Create splits preserving temporal order
    train_x = x_data[:train_idx].copy()
    train_y = y_data[:train_idx].copy()
    
    val_x = x_data[train_idx:val_idx].copy()
    val_y = y_data[train_idx:val_idx].copy()

    test_x = x_data[val_idx:].copy()
    test_y = y_data[val_idx:].copy()
    
    # Get sizes for split_info
    train_size = len(train_x)
    val_size = len(val_x)
    
    logger.info(f"\nTemporal split details:")
    logger.info(f"Training period: samples 0 to {train_idx - 1}")
    logger.info(f"Validation period: samples {train_idx} to {val_idx - 1}")
    logger.info(f"Testing period: samples {val_idx} to end")
    logger.info(f"Sequence window size: {n_pre}")
    
    return train_x, train_y, val_x, val_y, test_x, test_y, train_size, val_size
    
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
    """Log metrics to wandb if available.

    Args:
        history: Training history object
        start_time: Training start time
    """
    if not wandb_available:
        logger.info("wandb not available, skipping metrics logging")
        return

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

        # Print key metrics to console regardless of wandb availability
        logger.info(f"Training completed in {metrics_to_log['training_time']:.2f} seconds")
        if 'best_loss' in metrics_to_log:
            logger.info(f"Best loss: {metrics_to_log['best_loss']:.4f} at epoch {metrics_to_log['best_epoch']}")
        if 'best_val_loss' in metrics_to_log:
            logger.info(f"Best validation loss: {metrics_to_log['best_val_loss']:.4f}")

        # Log to wandb if available
        wandb.log(metrics_to_log)

    except Exception as e:
        logger.error(f"Error collecting metrics: {str(e)}")
        logger.error(traceback.format_exc())

def setup_wandb(config, validation_steps=None, train_dataset=None):
    """Initialize WandB with configuration if available.

    Args:
        config (dict): WandB configuration dictionary
        validation_steps (int, optional): Number of validation steps
        train_dataset: Training dataset for batch logging

    Returns:
        tuple: (run, wandb_callback) if wandb is available, or (None, dummy_callback) if not
    """
    if not wandb_available:
        # Return dummy objects if wandb is not available
        logger.warning("wandb not available, returning dummy callback")
        dummy_callback = tf.keras.callbacks.Callback()
        return None, dummy_callback

    # Initialize wandb only if available
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
    """Fixed implementation of CustomCallback with optional wandb support"""

    def __init__(self, train_dataset, use_wandb=False):
        super().__init__()  # Properly initialize parent class
        self._train_dataset = train_dataset
        self._start_time = time.time()
        self._epoch_start_time = None
        self._current_model = None  # Use a different name to avoid conflicts
        self._use_wandb = use_wandb and wandb_available

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

        # Print epoch time to console regardless of wandb setting
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")

        # Skip wandb logging if not enabled
        if not self._use_wandb:
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

        # Skip batch logging if wandb not enabled
        if not self._use_wandb:
            return

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
            mode='min',
            min_delta=0.001  # Add minimum improvement threshold
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

def process_sequence_data(x_data, col):
    """Process sequence data from comma-separated string format."""
    sequences = []
    for seq_str in x_data[col]:
        try:
            # Split by comma and convert to numeric, keeping sequence structure
            values = [float(x.strip()) for x in str(seq_str).split(',') if x.strip()]
            sequences.append(values)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error parsing sequence: {seq_str}, error: {e}")
            sequences.append([0.0] * 12)  # Default length
            
    # Convert to numpy for stats
    sequence_array = np.array(sequences)
    logger.info(f"{col} sequence stats:")
    logger.info(f"  Shape: {sequence_array.shape}")
    logger.info(f"  Mean: {np.mean(sequence_array):.3f}")
    logger.info(f"  Max: {np.max(sequence_array):.3f}")
    logger.info(f"  Non-zero entries: {np.sum(sequence_array > 0)}")
    
    return sequences

def load_data_from_csv(input_file, output_file):
    """
    Load and preprocess input and output data from CSV files with improved sequence handling.
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
        sequence_cols = []

        # First pass: identify sequences
        for col in x_data.columns:
            if col != 'ID' and isinstance(x_data[col].iloc[0], str) and ',' in str(x_data[col].iloc[0]):
                sequence_cols.append(col)
                logger.info(f"Found sequence column: {col}")

        # Process sequences first
        sequence_data = {}
        for col in sequence_cols:
            sequence_data[col] = process_sequence_data(x_data, col)
            # Store sequence data back in DataFrame as string to preserve it
            x_data[col] = [','.join(map(str, seq)) for seq in sequence_data[col]]

        # Function to determine if a column is binary
        def is_binary_column(series):
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 2:
                vals = set(unique_vals)
                return vals.issubset({0, 1, True, False, "0", "1"})
            return False

        # Function to determine if a column is continuous
        def is_continuous_column(col_name, series):
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                unique_vals = numeric_series.dropna().unique()
                if len(unique_vals) < 3:
                    return False
                has_decimals = any(float(x) % 1 != 0 for x in unique_vals if not pd.isna(x))
                return has_decimals or len(unique_vals) > 10
            except:
                return False

        # Identify column types for non-sequence columns
        for col in x_data.columns:
            if col != 'ID' and col not in sequence_cols:
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
        logger.info(f"Sequence columns: {sequence_cols}")
        
        # Handle binary columns
        for col in binary_cols:
            try:
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
            except Exception as e:
                logger.error(f"Error processing continuous column {col}: {str(e)}")
                x_data[col] = 0.0
        
        # Convert non-sequence columns to float32
        float_cols = [col for col in x_data.columns if col not in sequence_cols and col != 'ID']
        if float_cols:
            try:
                x_data[float_cols] = x_data[float_cols].astype(np.float32)
            except Exception as e:
                logger.error(f"Error converting to float32: {str(e)}")
                for col in float_cols:
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
        
        # Store sequence data as attribute
        x_data.attrs['sequence_data'] = sequence_data
        
        return x_data, y_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Input file exists: {os.path.exists(input_file)}")
        logger.error(f"Output file exists: {os.path.exists(output_file)}")
        logger.error(traceback.format_exc())
        raise
        
def configure_gpu(policy=None):
    """Configure GPU settings with improved error handling and fallbacks."""
    try:
        # Clear existing sessions
        tf.keras.backend.clear_session()
        gc.collect()

        # Try to import mixed precision safely
        try:
            # First try the newer API
            from tensorflow.keras.mixed_precision import global_policy
            mixed_precision_available = True
        except ImportError:
            try:
                # Fallback to older API
                from tensorflow.keras.mixed_precision import experimental as mixed_precision
                mixed_precision_available = True
            except ImportError:
                logger.warning("Mixed precision API not available in this TensorFlow version")
                mixed_precision_available = False
        
        # Set mixed precision if policy is provided and API is available
        if policy is not None and mixed_precision_available:
            try:
                if 'set_global_policy' in dir(mixed_precision):
                    mixed_precision.set_global_policy(policy)
                else:
                    mixed_precision.set_policy(policy)
                logger.info(f"Mixed precision policy set to {policy}")
            except Exception as e:
                logger.warning(f"Could not set mixed precision policy: {e}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.info("No GPUs available, using CPU")
            return False
        
        # Configure memory growth for available GPUs
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Set memory growth for GPU: {gpu.name}")
            except Exception as e:
                logger.warning(f"Could not set memory growth for GPU {gpu.name}: {e}")
        
        logger.info(f"Successfully configured {len(gpus)} GPU(s)")
        return True
    
    except Exception as e:
        logger.error(f"GPU configuration failed: {e}")
        logger.error(traceback.format_exc())
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
   
def create_dataset(x_data, y_data, n_pre, batch_size, loss_fn, J, 
                  is_training=False, is_censoring=False, 
                  pos_weight=None, neg_weight=None):
    logger.info(f"Creating dataset with parameters:")
    logger.info(f"n_pre: {n_pre}, batch_size: {batch_size}, loss_fn: {loss_fn}, J: {J}")
    logger.info(f"Input shape at start - features: {x_data.shape[1]}")

    # Define model features
    a_model_features = ['V3', 'white', 'black', 'latino', 'other', 
                       'mdd', 'bipolar', 'schiz', 'L1', 'L2', 'L3']

    # Remove ID if present
    if 'ID' in x_data.columns:
        x_data = x_data.drop('ID', axis=1)
    
    # Ensure required columns exist
    required_cols = ['V3', 'white', 'black', 'latino', 'other', 'mdd', 'bipolar', 'schiz']
    for col in ['L1', 'L2', 'L3']:
        if col not in x_data.columns:
            # Extract columns for this feature
            time_cols = [c for c in x_data.columns if c.startswith(f"{col}.")]
            if time_cols:
                # Combine time series into single column
                x_data[col] = x_data[time_cols].apply(lambda x: ','.join(map(str, x)), axis=1)
            else:
                logger.info(f"No time series found for {col}, using default values")
                x_data[col] = '0'  # Default value
    
    # Use required columns
    x_data = x_data[required_cols + ['L1', 'L2', 'L3']].copy()
    
    # Process L1, L2, L3 sequences
    sequence_features = {}
    for seq_col in ['L1', 'L2', 'L3']:
        sequences = []
        if 'sequence_data' in x_data.attrs and seq_col in x_data.attrs['sequence_data']:
        # Get pre-processed sequences
            sequences = x_data.attrs['sequence_data'][seq_col]
            # Convert to numpy array and ensure correct dimensionality
            seq_array = np.array(sequences)
            # If sequence is multi-dimensional, take first dimension
            if seq_array.ndim > 2:
                seq_array = seq_array.reshape(seq_array.shape[0], -1)
            sequence_features[seq_col] = seq_array
            logger.info(f"Using pre-processed {seq_col} sequences")
        else:
            # Process sequences from strings
            logger.info(f"Processing {seq_col} sequences from strings")
            for seq_str in x_data[seq_col]:
                try:
                    if isinstance(seq_str, str):
                        values = [float(x.strip()) for x in seq_str.split(',') if x.strip()]
                    elif isinstance(seq_str, (list, np.ndarray)):
                        values = [float(x) for x in seq_str if x is not None]
                    else:
                        values = [0.0] * n_pre
                        
                    if not values:
                        values = [0.0] * n_pre
                    elif len(values) < n_pre:
                        values = values + [values[-1]] * (n_pre - len(values))
                    sequences.append(values[:n_pre])
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error parsing {seq_col} sequence: {seq_str}, error: {e}")
                    sequences.append([0.0] * n_pre)
            
            sequence_features[seq_col] = np.array(sequences)

        # Log sequence stats
        logger.info(f"{seq_col} sequence stats:")
        logger.info(f"  Shape: {sequence_features[seq_col].shape}")
        logger.info(f"  Mean: {np.mean(sequence_features[seq_col]):.3f}")
        logger.info(f"  Max: {np.max(sequence_features[seq_col]):.3f}")
        logger.info(f"  Non-zero entries: {np.sum(sequence_features[seq_col] > 0)}")
    
    # Process non-sequence features
    non_seq_cols = [col for col in a_model_features if col not in ['L1', 'L2', 'L3']]
    x_values = x_data[non_seq_cols].values.astype(np.float32)
    
    # Scale continuous features
    cont_cols = ['V3']
    cont_indices = [non_seq_cols.index(col) for col in cont_cols if col in non_seq_cols]
    
    if cont_indices:
        for idx in cont_indices:
            col_values = x_values[:, idx]
            q75, q25 = np.percentile(col_values, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1.0
            median = np.median(col_values)
            x_values[:, idx] = (col_values - median) / (iqr + 1e-6)

    # Scale sequence features
    for seq_col in ['L1', 'L2', 'L3']:
        seq_values = sequence_features[seq_col]
        if np.any(seq_values != 0):
            if seq_col == 'L1':  # Special handling for L1
                # Handle non-positive values before log1p
                min_val = np.min(seq_values)
                if min_val <= 0:
                    seq_values = seq_values - min_val + 1e-6  # Shift to positive
                seq_transformed = np.log1p(seq_values)
            else:
                seq_transformed = seq_values.copy()  # Use copy to avoid modifying original
            
            # Handle constant sequences
            if np.std(seq_transformed) > 1e-6:  # Use small threshold
                q75, q25 = np.percentile(seq_transformed, [75, 25])
                iqr = q75 - q25 if q75 > q25 else 1.0
                median = np.median(seq_transformed)
                sequence_features[seq_col] = (seq_transformed - median) / (iqr + 1e-6)
            else:
                sequence_features[seq_col] = np.zeros_like(seq_transformed)

    # Process targets
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
            
            # Process targets for Y model
            if not is_censoring:  # Process for Y model
                # Get the minimum length to ensure arrays are aligned
                min_length = min(len(x_values), len(y_values))
                for seq_col in sequence_features:
                    min_length = min(min_length, len(sequence_features[seq_col]))
                
                # Truncate all arrays to the minimum length
                x_values = x_values[:min_length]
                y_values = y_values[:min_length]
                for seq_col in ['L1', 'L2', 'L3']:
                    sequence_features[seq_col] = sequence_features[seq_col][:min_length]
                
                if is_training:  # Only filter during training
                    # Now create and apply the mask
                    valid_mask = y_values != -1
                    logger.info(f"Found {np.sum(~valid_mask)} censored values out of {min_length}")
                    
                    # Apply mask to all arrays
                    x_values = x_values[valid_mask]
                    y_values = y_values[valid_mask]
                    for seq_col in ['L1', 'L2', 'L3']:
                        sequence_features[seq_col] = sequence_features[seq_col][valid_mask]
                    
                    y_values = (y_values > 0).astype(np.float32)
                else:  
                    # During testing/inference, keep -1 for censored values
                    y_values = y_values.astype(np.float32)
                    # Count statistics
                    valid_mask = y_values != -1
                    valid_values = y_values[valid_mask]
                    logger.info(f"Y value distribution:")
                    logger.info(f"  Total samples: {len(y_values)}")
                    logger.info(f"  Censored (-1): {np.sum(~valid_mask)}")
                    logger.info(f"  Valid: {np.sum(valid_mask)}")
                    if len(valid_values) > 0:
                        logger.info(f"  Valid distribution: {np.bincount(valid_values.astype(int))}")

                logger.info(f"Target distribution after sequence processing:")
                vals, counts = np.unique(y_values, return_counts=True)
                for v, c in zip(vals, counts):
                    logger.info(f"  {v}: {c}")
                
                logger.info(f"After processing - Target distribution:")
                logger.info(f"  0s: {np.sum(y_values == 0)}")
                logger.info(f"  1s: {np.sum(y_values == 1)}")
                logger.info(f"Final array shapes:")
                logger.info(f"  x_values: {x_values.shape}")
                logger.info(f"  y_values: {y_values.shape}")
                for seq_col in ['L1', 'L2', 'L3']:
                    logger.info(f"  {seq_col}: {sequence_features[seq_col].shape}")
        else:
            if all(f'A{i}' in y_data.columns for i in range(J)):
                y_values = y_data[[f'A{i}' for i in range(J)]].values
            else:
                raise ValueError("No suitable target columns found")

    # Create sequences for temporal forecasting
    num_samples = len(x_values) - n_pre
    
    # Calculate total feature dimension
    base_features = len(non_seq_cols)      # Non-sequence features
    seq_features = 3                       # L1, L2, L3
    time_features = 1                      # Global time index
    pos_features = 1                       # Position index
    total_features = base_features + seq_features + time_features + pos_features
    
    # Initialize arrays with correct dimensions
    x_sequences = np.zeros((num_samples, n_pre, total_features), dtype=np.float32)
    
    if loss_fn == "sparse_categorical_crossentropy":
        y_sequences = np.zeros(num_samples, dtype=np.int32)
    else:
        if len(y_values.shape) > 1:
            y_sequences = np.zeros((num_samples, y_values.shape[1]), dtype=np.float32)
        else:
            y_sequences = np.zeros((num_samples, 1), dtype=np.float32)

    # Add relative position features
    pos_index = np.linspace(0, 1, n_pre)[:, np.newaxis].astype(np.float32)

    # Fill sequences preserving temporal order
    for i in range(num_samples):
        seq_start = i
        seq_end = i + n_pre
        target_idx = seq_end - 1
        
        # Get base features
        current_features = x_values[seq_start:seq_end]
        
        # Process base features
        current_features = x_values[seq_start:seq_end]  # Shape: (n_pre, base_features)
        
        # Process sequence features with proper reshaping
        seq_values = []
        for seq_col in ['L1', 'L2', 'L3']:
            # Get one sequence window
            seq_window = sequence_features[seq_col][seq_start:seq_end]
            
            # Handle multi-dimensional sequences
            if seq_window.ndim > 1:
                # If sequence is already multi-dimensional, take the appropriate window
                window_size = min(n_pre, seq_window.shape[0])
                seq_window = seq_window[:window_size, 0]  # Take first dimension if multiple exist
                
                # Pad if necessary
                if window_size < n_pre:
                    pad_size = n_pre - window_size
                    seq_window = np.pad(seq_window, (0, pad_size), mode='edge')
            
            # Ensure 1D array of correct length
            if len(seq_window) != n_pre:
                # Truncate or pad to match n_pre
                if len(seq_window) > n_pre:
                    seq_window = seq_window[:n_pre]
                else:
                    seq_window = np.pad(seq_window, (0, n_pre - len(seq_window)), mode='edge')
            
            # Reshape to (n_pre, 1)
            seq_window = seq_window.reshape(n_pre, 1)
            seq_values.append(seq_window)
        
        # Create time features
        global_time = (np.arange(seq_start, seq_end) / len(x_values)).reshape(n_pre, 1).astype(np.float32)
        
        # Debug shapes
        if i == 0:
            logger.info(f"Shape check at first iteration:")
            logger.info(f"  current_features: {current_features.shape}")
            logger.info(f"  seq_values[0]: {seq_values[0].shape}")
            logger.info(f"  global_time: {global_time.shape}")
            logger.info(f"  pos_index: {pos_index.shape}")
        
        # Combine all features
        try:
            combined = np.concatenate(
                [current_features] +    # Base features
                seq_values +            # L1, L2, L3 sequences
                [global_time] +         # Time index
                [pos_index],            # Position index
                axis=1
            )
            x_sequences[i] = combined
        except Exception as e:
            logger.error(f"Error at iteration {i}:")
            logger.error(f"  Expected shape: {x_sequences[i].shape}")
            logger.error(f"  Got shapes: current_features={current_features.shape}, "
                      f"seq_values={[s.shape for s in seq_values]}, "
                      f"global_time={global_time.shape}, "
                      f"pos_index={pos_index.shape}")
            raise
        
        # Set target
        if target_idx < len(y_values):
            if loss_fn == "sparse_categorical_crossentropy":
                y_sequences[i] = y_values[target_idx]
            elif len(y_values.shape) > 1:
                y_sequences[i] = y_values[target_idx]
            else:
                y_sequences[i, 0] = y_values[target_idx]

    # Final preprocessing
    x_sequences = np.nan_to_num(x_sequences, nan=0.0, posinf=1.0, neginf=-1.0)

    logger.info(f"Final sequence shapes:")
    logger.info(f"X sequences: {x_sequences.shape}")
    logger.info(f"Y sequences: {y_sequences.shape}")
    logger.info(f"X range: [{np.min(x_sequences):.3f}, {np.max(x_sequences):.3f}]")
    logger.info(f"Final feature dimension: {x_sequences.shape[2]}")

    # Create dataset
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

    # Configure dataset
    if is_training:
        buffer_size = min(batch_size * 3, num_samples)
        dataset = (dataset
                  .take(num_samples)  # Take before cache
                  .cache()
                  .shuffle(buffer_size=buffer_size, seed=42)
                  .batch(batch_size, drop_remainder=is_training)
                  .repeat()
                  .prefetch(tf.data.AUTOTUNE))
    else:
        dataset = (dataset
                  .take(num_samples)  # Take before cache
                  .cache()
                  .batch(batch_size, drop_remainder=is_training)
                  .repeat()
                  .prefetch(tf.data.AUTOTUNE))

    return dataset, num_samples

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

def create_masked_loss(loss_fn, gbound=None, ybound=None):
    """Create a masked version of the specified loss function."""
    if loss_fn == "binary_crossentropy":
        return masked_binary_crossentropy
    else:
        return masked_sparse_categorical_crossentropy

def create_masked_metric(metric_fn):
    """Create a masked version of a metric that ignores censored values (-1)."""
    
    metric_name = metric_fn.name
    
    # Create the masked metric with appropriate name
    if isinstance(metric_fn, tf.keras.metrics.SparseCategoricalAccuracy):
        @tf.function
        def masked_metric(y_true, y_pred):
            mask = tf.not_equal(y_true, -1)
            mask = tf.cast(mask, tf.bool)
            indices = tf.where(mask)
            y_true_valid = tf.gather_nd(y_true, indices)
            y_pred_valid = tf.gather_nd(y_pred, indices)
            
            return tf.cond(
                tf.equal(tf.size(y_true_valid), 0),
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: metric_fn(y_true_valid, y_pred_valid)
            )
        
        # Set name after function definition
        masked_metric._name = 'masked_accuracy'
        masked_metric.__name__ = 'masked_accuracy'
        return masked_metric
        
    elif isinstance(metric_fn, tf.keras.metrics.SparseCategoricalCrossentropy):
        @tf.function
        def masked_metric(y_true, y_pred):
            mask = tf.not_equal(y_true, -1)
            mask = tf.cast(mask, tf.bool)
            indices = tf.where(mask)
            y_true_valid = tf.gather_nd(y_true, indices)
            y_pred_valid = tf.gather_nd(y_pred, indices)
            
            return tf.cond(
                tf.equal(tf.size(y_true_valid), 0),
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: metric_fn(y_true_valid, y_pred_valid)
            )
            
        # Set name after function definition
        masked_metric._name = 'masked_cross_entropy'
        masked_metric.__name__ = 'masked_cross_entropy'
        return masked_metric
        
    else:
        # Get appropriate name for binary metrics
        metric_display_name = 'masked_' + (
            'accuracy' if isinstance(metric_fn, tf.keras.metrics.BinaryAccuracy) else
            'auc' if isinstance(metric_fn, tf.keras.metrics.AUC) else
            'precision' if isinstance(metric_fn, tf.keras.metrics.Precision) else
            'recall' if isinstance(metric_fn, tf.keras.metrics.Recall) else
            'cross_entropy' if isinstance(metric_fn, tf.keras.metrics.BinaryCrossentropy) else
            metric_name
        )
        
        @tf.function
        def masked_metric(y_true, y_pred):
            # Ensure tensors have proper shape
            y_true = tf.convert_to_tensor(y_true)
            y_pred = tf.convert_to_tensor(y_pred)
            
            # Create mask
            mask = tf.not_equal(y_true, -1)
            mask = tf.cast(mask, tf.bool)
            
            # Ensure all inputs have rank 2
            y_true = tf.reshape(y_true, [-1, 1])
            y_pred = tf.reshape(y_pred, [-1, 1])
            mask = tf.reshape(mask, [-1, 1])
            
            # Apply mask
            y_true_masked = tf.where(mask, y_true, tf.zeros_like(y_true))
            y_pred_masked = tf.where(mask, y_pred, tf.zeros_like(y_pred))
            
            denominator = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
            result = metric_fn(y_true_masked, y_pred_masked)
            
            return result * tf.reduce_sum(tf.cast(mask, tf.float32)) / denominator
        
        # Set name after function definition
        masked_metric._name = metric_display_name
        masked_metric.__name__ = metric_display_name
        return masked_metric
        
def create_model(input_shape, output_dim, lr, dr, n_hidden, hidden_activation, 
                out_activation, loss_fn, J, epochs, steps_per_epoch, y_data=None, strategy=None, is_censoring=False, gbound=None, ybound=None):
    logger.info(f"Model input shape: {input_shape}")
    if strategy is None:
        strategy = get_strategy()

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
            'kernel_regularizer': l2(0.0005),
            'recurrent_regularizer': l2(0.0005),
            'bias_regularizer': l2(0.0005),
            'dropout': dr,
            'recurrent_dropout': 0,
            'unit_forget_bias': True,
            'dtype': tf.float32
        }

        # Use the MultiHeadAttention class defined earlier instead of duplicating code

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
            kernel_regularizer=l2(0.0005),
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

            # Calculate bias initialization
            init_bias = 0.0
            if y_data is not None:
                if 'target' in y_data.columns:
                    # For binary case
                    if output_units == 1:
                        valid_mask = y_data['target'].values != -1  # Exclude censored values
                        pos_ratio = float(np.mean(y_data['target'].values[valid_mask] > 0.5))
                        init_bias = np.log(pos_ratio / (1.0 - pos_ratio))
                        
                        # Single binary case metrics
                        metrics = [
                            create_masked_metric(tf.keras.metrics.BinaryAccuracy(name='accuracy')),
                            create_masked_metric(tf.keras.metrics.AUC(name='auc', multi_label=False)),
                            create_masked_metric(tf.keras.metrics.BinaryCrossentropy(name='cross_entropy'))
                        ]
                    else:
                        # Multi-label case (one-hot)
                        onehot_cols = [f'A{i}' for i in range(output_units)]
                        if all(col in y_data.columns for col in onehot_cols):
                            init_bias = []
                            for col in onehot_cols:
                                valid_mask = y_data[col] != -1
                                pos_ratio = float(np.mean(y_data[col][valid_mask]))
                                init_bias.append(np.log(pos_ratio / (1.0 - pos_ratio)))
                        
                        # Multi-label metrics
                        metrics = [
                            create_masked_metric(tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')),
                            create_masked_metric(tf.keras.metrics.SparseCategoricalCrossentropy(name='cross_entropy'))
                        ]

            loss = masked_binary_crossentropy if ybound is None else create_masked_loss("binary_crossentropy", ybound=ybound)

            metrics = [
                create_masked_metric(tf.keras.metrics.BinaryAccuracy(name='accuracy')),
                create_masked_metric(tf.keras.metrics.AUC(name='auc', multi_label=True if output_units > 1 else False)),
                create_masked_metric(tf.keras.metrics.Precision(name='precision')),
                create_masked_metric(tf.keras.metrics.Recall(name='recall')),
                create_masked_metric(tf.keras.metrics.BinaryCrossentropy(name='cross_entropy'))
        ]
        else:  # sparse_categorical_crossentropy
            final_activation = 'softmax'
            output_units = J

            # For categorical case, calculate class frequencies
            if y_data is not None and 'target' in y_data.columns:
                valid_mask = y_data['target'].values != -1
                class_counts = np.bincount(y_data['target'].values[valid_mask].astype(int), 
                                         minlength=J)
                class_probs = class_counts / np.sum(class_counts)
                init_bias = np.log(class_probs + 1e-7)  # Add small epsilon for numerical stability
            else:
                init_bias = np.zeros(J)
            
            loss = masked_sparse_categorical_crossentropy if gbound is None else create_masked_loss("sparse_categorical_crossentropy", gbound=gbound)
            metrics = [
                create_masked_metric(tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')),
                create_masked_metric(tf.keras.metrics.SparseCategoricalCrossentropy(name='cross_entropy'))
            ]
      
        # Create outputs layer
        outputs = tf.keras.layers.Dense(
            units=output_units,
            activation=final_activation,
            kernel_initializer='glorot_uniform',
            bias_initializer=tf.keras.initializers.Constant(init_bias),
            kernel_regularizer=l2(0.0005),
            name="output_dense"
        )(x)

        if outputs is None:
            raise ValueError("Outputs layer was not properly created. Check loss function configuration.")

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='lstm_model')

        # Learning rate schedule with longer warmup
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=steps_per_epoch * 8,  # Longer initial decay
            t_mul=1.5,  # Double period each restart
            m_mul=0.95,  # Slightly reduce max learning rate
            alpha=0.3  # Minimum learning rate
        )
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.0003,
            beta_1=0.92,
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
            weighted_metrics=None,
            jit_compile=True
        )
        
        return model