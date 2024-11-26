import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import LossScaleOptimizer  # Updated import

from utils import load_data_from_csv, create_dataset, configure_gpu, get_strategy

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

def test_model():
    global window_size, output_dir, nb_batches, loss_fn, J, is_censoring

    logger.info("Starting model testing...")
    
    # Configure GPU first
    if not configure_gpu(policy):
        logger.warning("GPU configuration failed, proceeding with default settings")
    
    try:
        # Load model and data
        model_path = os.path.join(output_dir, 
                                'trained_cat_model.h5' if loss_fn == "sparse_categorical_crossentropy" 
                                else 'trained_bin_model.h5')
        
        logger.info(f"Loading data from {output_dir}")
        x_data, y_data = load_data_from_csv(f"{output_dir}input_data.csv", f"{output_dir}output_data.csv")
        
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
        preds_test = model.predict(
            test_dataset,
            steps=test_steps,
            verbose=1
        )
        
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
        
        # Save predictions
        pred_file = os.path.join(output_dir, 
                                'test_cat_preds.npy' if loss_fn == "sparse_categorical_crossentropy"
                                else 'test_bin_preds.npy')
        np.save(pred_file, preds_test)
        logger.info(f"Test predictions saved to: {pred_file}")
        
        # Save detailed information
        info_path = os.path.join(output_dir, 'test_preds_info.npz')
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