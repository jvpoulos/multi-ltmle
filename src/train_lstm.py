import os

# Set critical environment variables before any TensorFlow imports
cuda_base = "/n/app/cuda/12.1-gcc-9.2.0"
os.environ.update({
    'CUDA_HOME': cuda_base,
    'CUDA_ROOT': cuda_base,
    'CUDA_PATH': cuda_base,
    'CUDNN_PATH': f"{cuda_base}/lib64/libcudnn.so",
    'LD_LIBRARY_PATH': f"{cuda_base}/lib64:{cuda_base}/extras/CUPTI/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    'PATH': f"{cuda_base}/bin:{os.environ.get('PATH', '')}",
    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
    'CUDA_VISIBLE_DEVICES': '0,1',
    'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
    'TF_XLA_FLAGS': '--tf_xla_enable_xla_devices',
    'XLA_FLAGS': f'--xla_gpu_cuda_data_dir={cuda_base}',
    'TF_GPU_THREAD_MODE': 'gpu_private',
    'TF_GPU_THREAD_COUNT': '2',
    'TF_CPP_MIN_LOG_LEVEL': '3'
})

import tensorflow as tf
import numpy as np
import pandas as pd
import time
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

# Suppress warnings and configure logging
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

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
        
    # Configure GPU
    print("\nChecking GPU configuration...")
    
    # Check CUDA module
    import subprocess
    try:
        module_list = subprocess.check_output('module list', shell=True).decode('utf-8')
        print("Loaded modules:")
        print(module_list)
    except:
        print("Could not check modules - may not be in module environment")

    # Check system CUDA
    try:
        nvcc_version = subprocess.check_output('nvcc --version', shell=True).decode('utf-8')
        print("\nNVCC Version:")
        print(nvcc_version)
    except:
        print("nvcc not found in PATH")

    # Check GPU devices
    try:
        nvidia_smi = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
        print("\nGPU Devices (nvidia-smi):")
        print(nvidia_smi)
    except:
        print("nvidia-smi not found!")

    # Configure GPU with detailed output
    gpu_available = configure_gpu(None)
    if gpu_available:
        print("\nGPU configuration successful")
        gpus = tf.config.list_physical_devices('GPU')
        for i, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            print(f"\nGPU {i} Name: {gpu.name}")
            print(f"GPU {i} Details: {details}")

        # Test GPU availability with a simple operation
        print("\nTesting GPU with simple operation...")
        with tf.device('/GPU:0'):
            try:
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print("GPU test successful. Matrix multiplication result:", c.numpy())
            except Exception as e:
                print(f"GPU test failed: {e}")
    else:
        print("\nGPU configuration failed, falling back to CPU")
        print("This will significantly impact performance")
        print("\nDebug information:")
        print(f"CUDA_HOME: {os.getenv('CUDA_HOME')}")
        print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")
        print(f"PATH: {os.getenv('PATH')}")

    # Get distribution strategy based on available devices
    strategy = get_strategy()
    logger.info(f"Using distribution strategy: {strategy.__class__.__name__}")
    
    # Log device info
    logical_devices = tf.config.list_logical_devices()
    logger.info("Available logical devices:")
    for device in logical_devices:
        logger.info(f"  {device.name} ({device.device_type})")

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
            # Check if outcome is Y and has binary loss function
            is_Y_outcome = any(col.startswith('Y') for col in (outcome_cols if isinstance(outcome_cols, list) else [outcome]))
            if is_Y_outcome and loss_fn == "binary_crossentropy":
                model_filename = 'lstm_bin_Y_model.h5'
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
        # Determine prediction filenames based on case
        if is_censoring:
            pred_filename = 'lstm_bin_C_preds.npy'
            info_filename = 'lstm_bin_C_preds_info.npz'
        else:
            # Check if outcome is Y and has binary loss function
            is_Y_outcome = any(col.startswith('Y') for col in (outcome_cols if isinstance(outcome_cols, list) else [outcome]))
            if is_Y_outcome and loss_fn == "binary_crossentropy":
                pred_filename = 'lstm_bin_Y_preds.npy'
                info_filename = 'lstm_bin_Y_preds_info.npz'
            else:
                if loss_fn == "sparse_categorical_crossentropy":
                    pred_filename = 'lstm_cat_A_preds.npy'
                    info_filename = 'lstm_cat_A_preds_info.npz'
                else:
                    pred_filename = 'lstm_bin_A_preds.npy'
                    info_filename = 'lstm_bin_A_preds_info.npz'

        # Set prediction and info paths
        pred_path = os.path.join(output_dir, pred_filename)
        info_path = os.path.join(output_dir, info_filename)

        # Save predictions
        np.save(pred_path, predictions)

        # Save prediction info
        np.savez(
            info_path,
            shape=predictions.shape,
            dtype=str(predictions.dtype),
            min_value=np.min(predictions),
            max_value=np.max(predictions),
            num_samples=n_valid_predictions,
            num_features=num_features
        )

        logger.info(f"Predictions saved to: {pred_path}")
        logger.info(f"Prediction info saved to: {info_path}")

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