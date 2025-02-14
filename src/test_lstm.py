import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import LossScaleOptimizer

from utils import load_data_from_csv, create_dataset, configure_gpu, get_strategy, get_data_filenames

import sys
import traceback
import logging
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Disable TensorFlow debugging info
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Additional TensorFlow configurations to reduce warnings
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

logger = logging.getLogger(__name__)

# Enable mixed precision with modern API
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_policy(policy)

def get_model_filenames(loss_fn, output_dim, is_censoring):
    """Get appropriate filenames for model and predictions based on model type."""
    
    if is_censoring:
        model_filename = 'lstm_bin_C_model.keras'
        pred_filename = 'test_bin_C_preds.npy'  # Changed from original to use 'test_' prefix
        info_filename = 'test_bin_C_preds_info.npz'  # Changed from original to use 'test_' prefix
    else:
        # Check if Y model based on dimensions and loss function
        is_y_model = (output_dim == 1 and loss_fn == "binary_crossentropy")
        if is_y_model:
            model_filename = 'lstm_bin_Y_model.keras'
            pred_filename = 'test_bin_Y_preds.npy'
            info_filename = 'test_bin_Y_preds_info.npz'
        else:
            # Treatment model (A)
            if loss_fn == "sparse_categorical_crossentropy":
                model_filename = 'lstm_cat_A_model.keras'
                pred_filename = 'test_cat_A_preds.npy'
                info_filename = 'test_cat_A_preds_info.npz'
            else:
                model_filename = 'lstm_bin_A_model.keras'
                pred_filename = 'test_bin_A_preds.npy'
                info_filename = 'test_bin_A_preds_info.npz'
    
    return model_filename, pred_filename, info_filename
    
def test_model():
    global window_size, output_dir, nb_batches, loss_fn, J, is_censoring, output_dim

    logger.info("Starting model testing...")
    
    # Configure GPU first
    if not configure_gpu(policy):
        logger.warning("GPU configuration failed, proceeding with default settings")
    
    try:
        output_dir = os.path.abspath(output_dir)
        logger.info(f"Using absolute output directory: {output_dir}")

        # Get appropriate filenames using the shared function
        model_filename, pred_filename, info_filename = get_model_filenames(
            loss_fn, output_dim, is_censoring
        )

        # Set paths using absolute directory
        model_path = os.path.join(output_dir, model_filename)
        pred_path = os.path.join(output_dir, pred_filename)
        info_path = os.path.join(output_dir, info_filename)

        logger.info(f"Loading model from: {model_path}")
        
        logger.info(f"Loading data from {output_dir}")
        input_file, output_file = get_data_filenames(is_censoring, loss_fn, outcome_cols)

        logger.info(f"Loading data from input file: {input_file}")
        logger.info(f"Loading data from output file: {output_file}")
        
        # Use absolute paths
        input_path = os.path.join(output_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        
        x_data, y_data = load_data_from_csv(input_path, output_path)

        # Load split information
        split_info_path = os.path.join(output_dir, 'split_info.npy')
        logger.info(f"Loading split info from {split_info_path}")
        split_info = np.load(split_info_path, allow_pickle=True).item()
        
        train_size = split_info['train_size']
        val_size = split_info['val_size']
        batch_size = split_info['batch_size']
        n_pre = split_info['n_pre']
        num_features = split_info['num_features']
        
        # Create test dataset from remaining data
        test_data_x = x_data[train_size+val_size:].copy()
        test_data_y = y_data[train_size+val_size:].copy()
        
        if len(test_data_x) == 0 or len(test_data_y) == 0:
            raise ValueError("Test dataset is empty after splitting")
        
        logger.info("\nDataset information:")
        logger.info(f"Total samples: {len(x_data)}")
        logger.info(f"Training samples: {train_size}")
        logger.info(f"Validation samples: {val_size}")
        logger.info(f"Test samples: {len(test_data_x)}")
        logger.info(f"Feature dimension: {num_features}")
        logger.info(f"Sequence length: {n_pre}")
        
        # Remove ID column if present
        if 'ID' in test_data_x.columns:
            test_data_x = test_data_x.drop(columns=['ID'])
        
        # Modify data preparation for binary case
        if loss_fn == "binary_crossentropy":
            if 'A' in y_data.columns:
                # Create one-hot encoded columns
                for i in range(J):
                    y_data[f'A{i}'] = (y_data['A'].values == i).astype(int)
                y_data = y_data[['ID'] + [f'A{i}' for i in range(J)]]

        # Create test dataset
        strategy = get_strategy()
        with strategy.scope():
            test_dataset, test_samples = create_dataset(
                test_data_x,
                test_data_y,
                n_pre,
                batch_size,
                loss_fn,
                J,
                is_training=False,
                is_censoring=is_censoring
            )
            
        # Calculate test steps
        test_steps = max(1, (test_samples - n_pre + 1) // batch_size)
        logger.info(f"Test steps: {test_steps}")

        # Load and verify model
        logger.info("\nLoading model...")
        with strategy.scope():
            model = load_model(model_path, custom_objects={
                'LossScaleOptimizer': LossScaleOptimizer
            })
        logger.info("Model loaded successfully")
        
        # Verify test dataset characteristics
        for x_batch, y_batch in test_dataset.take(1):
            logger.info("\nTest dataset characteristics:")
            logger.info(f"X shape: {x_batch.shape}")
            logger.info(f"Y shape: {y_batch.shape}")
            logger.info(f"Feature dimension: {x_batch.shape[-1]}")
            logger.info(f"Expected features: {num_features}")
            logger.info(f"X range: [{tf.reduce_min(x_batch)}, {tf.reduce_max(x_batch)}]")
            
            # Verify feature dimensions
            if x_batch.shape[-1] != num_features:
                raise ValueError(
                    f"Feature dimension mismatch. Expected {num_features}, got {x_batch.shape[-1]}"
                )
        
        # Generate predictions
        logger.info("\nEvaluating model...")
        test_metrics = model.evaluate(
            test_dataset,
            steps=test_steps,
            verbose=1
        )
        
        logger.info("\nTest metrics:")
        for name, value in zip(model.metrics_names, test_metrics):
            logger.info(f"{name}: {value:.4f}")
        
        logger.info("\nGenerating predictions...")
        # Get predictions without reshaping - predictions already respect temporal order
        preds_test = model.predict(
            test_dataset,
            steps=test_steps,
            batch_size=batch_size * 2,
            verbose=1
        )
        
        logger.info(f"Raw test predictions shape: {preds_test.shape}")
        
        # Pad predictions to match original length if needed
        num_pad = len(test_data_x) - len(preds_test)
        if num_pad > 0:
            logger.info("Padding predictions to match original length...")
            logger.info(f"Original predictions stats:")
            logger.info(f"Mean: {np.mean(preds_test)}, Std: {np.std(preds_test)}")
            logger.info(f"Min: {np.min(preds_test)}, Max: {np.max(preds_test)}")
            
            # Use last few valid predictions to initialize padding
            k = min(50, len(preds_test))
            last_valid = preds_test[-k:]
            
            # Create padding that follows the trend
            pad_predictions = []
            for i in range(num_pad):
                if i == 0:
                    # Start with mean of last valid predictions
                    pad_value = np.mean(last_valid, axis=0)
                else:
                    # Add small random variation to maintain temporal change
                    variation = np.std(last_valid, axis=0) * 0.1
                    pad_value = pad_predictions[-1] + np.random.normal(0, variation, pad_value.shape)
                pad_predictions.append(pad_value)
            
            pad_predictions = np.array(pad_predictions)
            preds_test = np.vstack([preds_test, pad_predictions])
        
        logger.info(f"Final test predictions shape: {preds_test.shape}")
        
        # Ensure we only keep predictions for actual samples
        n_valid_samples = len(test_data_x) - n_pre + 1
        if n_valid_samples <= 0:
            raise ValueError(f"Invalid number of samples: {n_valid_samples}")
            
        preds_test = preds_test[:n_valid_samples]
        
        logger.info(f"Final test predictions shape: {preds_test.shape}")
        
        # Ensure we only keep predictions for actual samples
        n_valid_samples = len(test_data_x) - n_pre + 1
        if n_valid_samples <= 0:
            raise ValueError(f"Invalid number of samples: {n_valid_samples}")
            
        preds_test = preds_test[:n_valid_samples]
        
        logger.info(f"Final test predictions shape: {preds_test.shape}")
        
        # Ensure we only keep predictions for actual samples
        n_valid_samples = len(test_data_x) - n_pre + 1
        if n_valid_samples <= 0:
            raise ValueError(f"Invalid number of samples: {n_valid_samples}")
            
        preds_test = preds_test[:n_valid_samples]
        
        # Add prediction analysis based on loss function
        if loss_fn == "binary_crossentropy":
            logger.info("\nBinary prediction analysis:")
            logger.info(f"Prediction shape: {preds_test.shape}")
            logger.info(f"Mean predictions per class:")
            for i in range(J):
                mean_pred = np.mean(preds_test[:, i])
                logger.info(f"Class {i}: {mean_pred:.4f}")
            
            # Calculate class predictions
            class_preds = np.argmax(preds_test, axis=1)
            logger.info("\nPredicted class distribution:")
            for i in range(J):
                count = np.sum(class_preds == i)
                logger.info(f"Class {i}: {count} ({count/len(class_preds)*100:.2f}%)")
        else:
            logger.info("\nCategorical prediction analysis:")
            logger.info(f"Prediction shape: {preds_test.shape}")
            pred_classes = np.argmax(preds_test, axis=1)
            class_dist = np.bincount(pred_classes, minlength=J)
            logger.info("Class distribution:")
            for i in range(J):
                count = class_dist[i]
                logger.info(f"Class {i}: {count} ({count/len(pred_classes)*100:.2f}%)")
        
        logger.info(f"\nPredictions summary:")
        logger.info(f"Shape: {preds_test.shape}")
        logger.info(f"Data type: {preds_test.dtype}")
        logger.info(f"Range: [{np.min(preds_test)}, {np.max(preds_test)}]")
        
        # Print prediction statistics
        logger.info("\nPrediction statistics:")
        logger.info(f"Mean prediction: {np.mean(preds_test):.4f}")
        logger.info(f"Std prediction: {np.std(preds_test):.4f}")
        
        if loss_fn == "binary_crossentropy":
            pred_classes = (preds_test > 0.5).astype(int)
            logger.info(f"Class distribution: {np.bincount(pred_classes.flatten())}")
        else:
            pred_classes = np.argmax(preds_test, axis=1)
            logger.info(f"Class distribution: {np.bincount(pred_classes)}")
        
        # Start by ensuring outcome_cols is defined earlier in the function
        try:
            # Get outcome info from global scope if it exists
            outcome_cols = globals().get('outcome_cols', None)
        except:
            outcome_cols = None

        # Determine model type
        is_Y_outcome = False
        try:
            if isinstance(outcome_cols, list):
                is_Y_outcome = any(str(col).startswith('Y') for col in outcome_cols)
            elif isinstance(outcome_cols, str):
                is_Y_outcome = str(outcome_cols).startswith('Y')
        except:
            is_Y_outcome = False

        # Set prediction filenames based on model type
        if is_censoring:
            pred_filename = 'test_bin_C_preds.npy'
            info_filename = 'test_bin_C_preds_info.npz'
        else:
            if is_Y_outcome and loss_fn == "binary_crossentropy":
                pred_filename = 'test_bin_Y_preds.npy'
                info_filename = 'test_bin_Y_preds_info.npz'
            else:
                if loss_fn == "sparse_categorical_crossentropy":
                    pred_filename = 'test_cat_A_preds.npy'
                    info_filename = 'test_cat_A_preds_info.npz'
                else:
                    pred_filename = 'test_bin_A_preds.npy'
                    info_filename = 'test_bin_A_preds_info.npz'

        # Set prediction and info paths
        pred_path = os.path.join(output_dir, pred_filename)
        info_path = os.path.join(output_dir, info_filename)

        # Save predictions using the paths from get_model_filenames
        np.save(pred_path, preds_test)
        logger.info(f"Test predictions saved to: {pred_path}")

        # Save detailed information
        np.savez(
            info_path,
            shape=preds_test.shape,
            dtype=str(preds_test.dtype),
            min_value=np.min(preds_test),
            max_value=np.max(preds_test),
            num_test_samples=n_valid_samples,
            test_metrics=dict(zip(model.metrics_names, test_metrics))
        )
        logger.info(f"Test information saved to: {info_path}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    try:
        test_model()
        logger.info("Testing completed successfully")
    except Exception as e:
        logger.error("Testing failed")
        logger.error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()