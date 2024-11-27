import os

# Set correct CUDA path
cuda_path = "/n/app/cuda/11.7-gcc-9.2.0"
os.environ.update({
    'CUDA_HOME': cuda_path,
    'LD_LIBRARY_PATH': f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    'PATH': f"{cuda_path}/bin:{os.environ.get('PATH', '')}",
    'XLA_FLAGS': f'--xla_gpu_cuda_data_dir={cuda_path}',
    'TF_XLA_FLAGS': '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
})

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

from utils import create_model, load_data_from_csv, create_dataset, get_optimized_callbacks, configure_gpu, get_strategy, setup_wandb, CustomCallback, log_metrics, FilterPatterns, CustomNanCallback

import sys
import traceback
import logging
import warnings

import wandb
from datetime import datetime

import gc
tf.keras.backend.clear_session()
gc.collect()

# Suppress all tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

# Create custom filter for TensorFlow messages
class TFMessageFilter(logging.Filter):
    def filter(self, record):
        return 'tensorflow' not in record.name.lower() and \
               'executor.cc' not in record.getMessage() and \
               'INVALID_ARGUMENT' not in record.getMessage()

# Apply filters to root logger and tensorflow logger
root_logger = logging.getLogger()
tf_logger = logging.getLogger('tensorflow')
message_filter = TFMessageFilter()
root_logger.addFilter(message_filter)
tf_logger.addFilter(message_filter)

# Suppress specific tensorflow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')

# Disable AutoGraph warnings
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

# Configure TensorFlow to suppress executor messages
tf.debugging.disable_traceback_filtering()

# Additional warning suppression environment variables
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'AUTOGRAPH_VERBOSITY': '0',
    'TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE': 'False',
    'TF_DISABLE_COMPILATION_NOTIFICATIONS': '1',
    'TF_ENABLE_ONEDNN_OPTS': '0',
    'TF_DISABLE_MKL': '1',
    'TF_ENABLE_GPU_GARBAGE_COLLECTION': 'False',
})

# Configure stdout/stderr to filter TensorFlow messages
class TFOutputFilter:
    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, text):
        if not any(msg in text for msg in [
            'tensorflow', 'INVALID_ARGUMENT', 'executor.cc',
            'GetNextFromShard', 'MultiDeviceIterator'
        ]):
            self.original_stream.write(text)

    def flush(self):
        self.original_stream.flush()

sys.stdout = TFOutputFilter(sys.stdout)
sys.stderr = TFOutputFilter(sys.stderr)

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def main():
    global n_pre, nb_batches, output_dir, loss_fn, epochs, lr, dr, n_hidden, hidden_activation, out_activation, patience, J, window_size, is_censoring

    # Record start time
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
        
    # Configure GPU with fallback
    gpu_available = configure_gpu(None)
    logger.info(f"Using {'GPU' if gpu_available else 'CPU'} for training")

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Loss function: {loss_fn}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Dropout rate: {dr}")
    logger.info(f"Hidden units: {n_hidden}")
    logger.info(f"Hidden activation: {hidden_activation}")
    logger.info(f"Output activation: {out_activation}")
    logger.info(f"Patience: {patience}")
    logger.info(f"J: {J}")
    logger.info(f"Window size: {window_size}")
    logger.info(f"Batch size: {nb_batches}")

    n_pre = int(window_size)
    batch_size = int(nb_batches)

    # Get distribution strategy
    strategy = get_strategy()
    logger.info(f"Using distribution strategy: {strategy.__class__.__name__}")

    # Load data
    x_data, y_data = load_data_from_csv(f"{output_dir}input_data.csv", f"{output_dir}output_data.csv")

    logger.info("Data loaded successfully")
    logger.info(f"x_data shape: {x_data.shape}")
    logger.info(f"y_data shape: {y_data.shape}")

    # Modify the y_data preparation section:
    if loss_fn == "binary_crossentropy":
        # For binary case, ensure we have one-hot encoded targets
        if 'A' in y_data.columns:
            # Get unique treatments and verify range
            unique_treatments = np.unique(y_data['A'].values)
            logger.info(f"Unique treatments: {unique_treatments}")
            
            # Create one-hot encoded columns
            for i in range(J):
                y_data[f'A{i}'] = (y_data['A'].values == i).astype(int)
            
            # Keep only ID and one-hot columns
            y_data = y_data[['ID'] + [f'A{i}' for i in range(J)]]
    else:
        # For categorical case, keep existing A column
        if 'A' not in y_data.columns and 'target' not in y_data.columns:
            raise ValueError("No target or A column found in y_data")

    logger.info("Data shapes after processing:")
    logger.info(f"x shape: {x_data.shape}")
    logger.info(f"y shape: {y_data.shape}")

    if y_data.empty:
        raise ValueError("y_data is empty after processing")

    # Calculate splits
    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)  # 10% for validation
    test_size = num_samples - train_size - val_size

    # Split data
    train_x = x_data[:train_size].copy()
    train_y = y_data[:train_size].copy()
    
    val_x = x_data[train_size:train_size+val_size].copy()
    val_y = y_data[train_size:train_size+val_size].copy()
    
    test_x = x_data[train_size+val_size:].copy()
    test_y = y_data[train_size+val_size:].copy()

    # Calculate feature dimension
    num_features = len(x_data.columns)
    if 'ID' in x_data.columns:
        num_features -= 1

    logger.info(f"\nDataset splits:")
    logger.info(f"Total samples: {num_samples}")
    logger.info(f"Training samples: {len(train_x)}")
    logger.info(f"Validation samples: {len(val_x)}")
    logger.info(f"Test samples: {len(test_x)}")
    logger.info(f"Number of features: {num_features}")
    logger.info(f"Feature columns: {x_data.columns.tolist()}")

    # Create datasets
    with strategy.scope():
        train_dataset, train_samples = create_dataset(
            train_x, train_y,
            n_pre,
            batch_size,
            loss_fn,
            J,
            is_training=True,
            is_censoring=is_censoring
        )
        val_dataset, val_samples = create_dataset(
            val_x, val_y,
            n_pre,
            batch_size,
            loss_fn,
            J,
            is_training=False,
            is_censoring=is_censoring
        )

    # Calculate steps
    steps_per_epoch = max(1, (train_samples - n_pre + 1) // batch_size)
    validation_steps = max(1, (val_samples - n_pre + 1) // batch_size)

    logger.info(f"Training steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")

    # Save split information
    split_info = {
        'train_size': train_size,
        'val_size': val_size,
        'batch_size': batch_size,
        'n_pre': n_pre,
        'num_features': num_features,
        'steps_per_epoch': steps_per_epoch,
        'validation_steps': validation_steps
    }
    np.save(os.path.join(output_dir, 'split_info.npy'), split_info)

    # Configure model
    input_shape = (n_pre, num_features)
    output_dim = J  # Use J for both cases now
    logger.info(f"Using {'censoring' if is_censoring else 'treatment'} prediction with output_dim={output_dim}")

    logger.info(f"\nModel configuration:")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output dimension: {output_dim}")

    # Update WandB config
    wandb_config = {
        'learning_rate': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'hidden_units': n_hidden,
        'dropout_rate': dr,
        'optimizer': 'Adam',
        'input_shape': (n_pre, num_features),
        'output_dim': output_dim,
        'loss_function': loss_fn,
        'hidden_activation': hidden_activation,
        'output_activation': out_activation,
        'training_samples': num_samples,
        'validation_samples': val_size,
        'steps_per_epoch': steps_per_epoch,
        'validation_steps': validation_steps,
        'window_size': window_size,
    }

    # Initialize WandB
    run, wandb_callback = setup_wandb(
        config=wandb_config,
        validation_steps=validation_steps,
        train_dataset=train_dataset
    )
    
    with strategy.scope():
        model = create_model(
            input_shape=input_shape,
            output_dim=output_dim,
            lr=lr,
            dr=dr,
            n_hidden=n_hidden,
            hidden_activation=hidden_activation,
            out_activation=out_activation,
            loss_fn=loss_fn,
            J=J,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            y_data=y_data,
            strategy=strategy,
            is_censoring=is_censoring
        )

    # Get callbacks
    callbacks = get_optimized_callbacks(patience, output_dir, train_dataset)
    callbacks.append(wandb_callback)
    callbacks.append(CustomCallback(train_dataset))
    callbacks.append(CustomNanCallback())

    try:
        # Train model with class weights for binary classification
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
        )

        # Log metrics with proper handling of binary case
        log_metrics(history, start_time)        

        # Determine model filename based on case
        if is_censoring:
            model_filename = 'lstm_bin_C_model.h5'
        else:
            if loss_fn == "sparse_categorical_crossentropy":
                model_filename = 'lstm_cat_A_model.h5'
            else:
                model_filename = 'lstm_bin_A_model.h5'

        # Set model path
        model_path = os.path.join(output_dir, model_filename)

        # Save model
        model.save(model_path)
        logger.info(f"Model saved to: {model_path}")

        # Log the SavedModel as a WandB Artifact
        artifact = wandb.Artifact('trained_model', type='model')
        artifact.add_file(model_path)
        run.log_artifact(artifact)
        
        # Generate final predictions
        logger.info("\nGenerating predictions for all data...")
        
        x_data_final = x_data.copy()
        if 'ID' in x_data_final.columns:
            x_data_final = x_data_final.drop(columns=['ID'])
        
        final_dataset, final_samples = create_dataset(
            x_data_final,
            y_data,
            n_pre,
            batch_size,
            loss_fn,
            J,
            is_training=False,
            is_censoring=output_dim==1
        )
        
        total_steps = max(1, (final_samples - n_pre + 1) // batch_size)
        predictions = model.predict(
            final_dataset,
            steps=total_steps,
            verbose=1
        )
        
        n_valid_predictions = len(x_data_final) - n_pre + 1
        predictions = predictions[:n_valid_predictions]
        
        # Save predictions
        pred_path = os.path.join(output_dir, 
                               'lstm_cat_preds.npy' if loss_fn == "sparse_categorical_crossentropy" 
                               else 'lstm_bin_preds.npy')
        np.save(pred_path, predictions)
        
        # Save prediction info
        info_path = os.path.join(output_dir, 
                               'lstm_cat_preds_info.npz' if loss_fn == "sparse_categorical_crossentropy" 
                               else 'lstm_bin_preds_info.npz')

        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Predictions type: {predictions.dtype}")
        logger.info(f"Sample predictions: {predictions[:5]}")
        
        np.savez(
            info_path,
            shape=predictions.shape,
            dtype=str(predictions.dtype),
            min_value=np.min(predictions),
            max_value=np.max(predictions),
            num_samples=n_valid_predictions,
            num_features=num_features
        )
        logger.info("Predictions saved successfully")

    except Exception as e:
        wandb.alert(
            title="Training Failed",
            text=str(e)
        )
        logger.error(f"Error during execution: {str(e)}")
        logger.error(f"Traceback:", exc_info=True)
        return
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()