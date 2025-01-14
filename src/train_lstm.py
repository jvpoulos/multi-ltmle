import os
import math

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

# Import necessary modules
import tensorflow as tf

# Configure TensorFlow for CPU
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)
    
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

from utils import create_model, load_data_from_csv, create_dataset, get_optimized_callbacks, configure_device, get_strategy, setup_wandb, CustomCallback, log_metrics, CustomNanCallback

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
        
    configure_device()
    strategy = get_strategy()
    
    # Log device information
    devices = tf.config.list_physical_devices()
    logger.info("Available devices:")
    for device in devices:
        logger.info(f"  {device.device_type}: {device.name}")
    
    # Remove the existing GPU configuration checks and replace with:
    if tf.config.list_physical_devices('GPU'):
        logger.info("Running with GPU support")
        # Log GPU memory info if available
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            for gpu in gpu_devices:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                logger.info(f"GPU Details: {gpu_details}")
        except:
            pass
    else:
        logger.info("Running on CPU")
        logger.info(f"Number of CPU cores: {os.cpu_count()}")
        logger.info(f"TensorFlow inter-op threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
        logger.info(f"TensorFlow intra-op threads: {tf.config.threading.get_intra_op_parallelism_threads()}")

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
        # For binary case, ensure proper one-hot encoding and class balance
        if 'A' in y_data.columns:
            # Get class distribution
            treatment_dist = y_data['A'].value_counts(normalize=True)
            logger.info(f"Original treatment distribution:\n{treatment_dist}")
            
            # Create one-hot encoded columns with proper handling
            onehot_cols = []
            for i in range(J):
                col_name = f'A{i}'
                y_data[col_name] = (y_data['A'].values == i).astype(float)
                # Add small noise to prevent perfect separation
                if len(y_data[y_data[col_name] == 1]) > 0:
                    noise = np.random.normal(0, 0.1, size=len(y_data))
                    y_data[col_name] = y_data[col_name] + noise * y_data[col_name]
                    y_data[col_name] = np.clip(y_data[col_name], 0.05, 0.95)
                onehot_cols.append(col_name)
                
            # Keep only ID and one-hot columns
            y_data = y_data[['ID'] + onehot_cols]
            
            # Log class balance after processing
            logger.info("Treatment distribution after processing:")
            for col in onehot_cols:
                mean_val = y_data[col].mean()
                logger.info(f"{col}: {mean_val:.4f}")
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

    # Calculate steps directly based on samples
    steps_per_epoch = math.ceil(train_samples / batch_size)
    validation_steps = math.ceil(val_samples / batch_size)

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

    if not is_censoring:
        if loss_fn == "binary_crossentropy":
            # For binary classification (Y model or single treatment)
            if output_dim == 1:
                if 'target' in y_data.columns:
                    counts = y_data['target'].value_counts(normalize=True)
                    # Calculate weights with smoothing
                    pos_weight = 1.0 / (counts.get(1, 0.5) + 1e-6)
                    neg_weight = 1.0 / (counts.get(0, 0.5) + 1e-6)
                    # Normalize
                    total = pos_weight + neg_weight
                    class_weight = {
                        0: neg_weight / total,
                        1: pos_weight / total
                    }
                    logger.info(f"Binary classification weights - Neg: {class_weight[0]:.4f}, Pos: {class_weight[1]:.4f}")
                else:
                    class_weight = {0: 0.5, 1: 0.5}
                    logger.info("Using default equal weights for binary classification")
            
            # For multi-class with binary crossentropy (one-hot)
            else:
                onehot_cols = [f'A{i}' for i in range(J)]
                if all(col in y_data.columns for col in onehot_cols):
                    # Calculate weights from one-hot columns with smoothing
                    class_sums = [y_data[col].sum() for col in onehot_cols]
                    total_samples = sum(class_sums)
                    
                    # Calculate balanced weights
                    class_weight = {
                        i: total_samples / (max(sum, 1) * J)
                        for i, sum in enumerate(class_sums)
                    }
                    
                    logger.info("One-hot encoded class weights:")
                    for i, w in class_weight.items():
                        logger.info(f"Class {i}: {w:.4f} (samples: {class_sums[i]})")
                
                elif 'target' in y_data.columns:
                    # Calculate from target column
                    counts = y_data['target'].value_counts()
                    total_samples = len(y_data)
                    
                    # Initialize weights for all classes
                    class_weight = {i: 1.0 for i in range(J)}
                    
                    # Update weights for observed classes
                    for class_idx, count in counts.items():
                        if 0 <= class_idx < J:  # Ensure valid class index
                            class_weight[class_idx] = total_samples / (count * J)
                    
                    logger.info("Target-based class weights:")
                    for i, w in sorted(class_weight.items()):
                        count = counts.get(i, 0)
                        logger.info(f"Class {i}: {w:.4f} (samples: {count})")
                
                else:
                    # Fallback to uniform weights
                    class_weight = {i: 1.0 for i in range(J)}
                    logger.info("Using uniform weights for all classes")
        
        else:
            # Categorical crossentropy case
            if 'target' in y_data.columns:
                counts = y_data['target'].value_counts()
                total_samples = len(y_data)
                
                # Calculate balanced weights for all classes
                class_weight = {}
                for i in range(J):
                    count = counts.get(i, 0)
                    # Add smoothing factor to prevent division by zero
                    class_weight[i] = total_samples / (max(count, 1) * J)
                
                logger.info("Categorical class weights:")
                for i, w in sorted(class_weight.items()):
                    count = counts.get(i, 0)
                    logger.info(f"Class {i}: {w:.4f} (samples: {count})")
            else:
                # Fallback to uniform weights
                class_weight = {i: 1.0 for i in range(J)}
                logger.info("Using uniform weights for categorical classes")

    if is_censoring:
        # Calculate class weights based on observed frequencies in y_data
        if 'target' in y_data.columns:
            target_vals = y_data['target'].values
            # Convert negative values to 0 and ensure binary classes
            target_vals = np.where(target_vals < 0, 0, target_vals)
            target_vals = np.where(target_vals > 0, 1, target_vals)
            
            c_counts = np.bincount(target_vals.astype(int))
            total = len(target_vals)
            
            # Calculate balanced weights
            class_weight = {
                0: total/(2 * max(c_counts[0], 1)),  # Add max to prevent div by 0
                1: total/(2 * max(c_counts[1], 1) if len(c_counts) > 1 else 1)
            }
            
            # Log censoring distribution
            logger.info("Censoring class distribution:")
            for i in [0, 1]:
                count = c_counts[i] if i < len(c_counts) else 0 
                pct = (count/total) * 100
                logger.info(f"Class {i}: {count} ({pct:.2f}%)")
        else:
            # Fallback to balanced weights if no target column
            class_weight = {0: 1.0, 1: 1.0}
            logger.info("Using balanced weights for censoring")
            
        # Adjust weights based on class imbalance
        min_class_weight = min(class_weight.values())
        if min_class_weight < 0.1:  # If severe imbalance
            # Scale up minority class but keep majority reasonable
            scale = 0.1 / min_class_weight
            class_weight = {k: min(v * scale, 10.0) for k,v in class_weight.items()}

        logger.info("Censoring model class weights:")
        for k,v in sorted(class_weight.items()):
            logger.info(f"Class {k}: {v:.4f}")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight,  # Add class weights
        verbose=1
    )

    # Log metrics with proper handling of binary case
    log_metrics(history, start_time)        

    # Determine model filename based on case
    if is_censoring:
        model_filename = 'lstm_bin_C_model.keras'
    else:
        # Check if Y model based on dimensions and loss function
        is_y_model = (output_dim == 1 and loss_fn == "binary_crossentropy")
        if is_y_model:
            model_filename = 'lstm_bin_Y_model.keras'
        else:
            # Treatment model (A)
            model_filename = 'lstm_cat_A_model.keras' if loss_fn == "sparse_categorical_crossentropy" else 'lstm_bin_A_model.keras'

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
    
    is_Y_outcome = any(col.startswith('Y') for col in outcome_cols if isinstance(outcome_cols, str))
    # OR if we need to handle both string and list cases
    if isinstance(outcome_cols, list):
        is_Y_outcome = any(col.startswith('Y') for col in outcome_cols)
    elif isinstance(outcome_cols, str):
        is_Y_outcome = outcome_cols.startswith('Y')
    else:
        is_Y_outcome = False

    # Use this to determine prediction filenames
    if is_censoring:
        pred_filename = 'lstm_bin_C_preds.npy'
        info_filename = 'lstm_bin_C_preds_info.npz'
    else:
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
    
    wandb.finish()

if __name__ == "__main__":
    main()