import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import LossScaleOptimizer  # Updated import

from utils import data_generator, load_data_from_csv, get_output_signature, create_dataset

import sys
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable mixed precision with modern API
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_policy(policy)

def test_model():
    global window_size, output_dir, nb_batches, loss_fn, J

    logger.info("Starting model testing...")
    
    try:
        # Load model and data
        model_path = os.path.join(output_dir, 
                                'trained_cat_model.h5' if loss_fn == "sparse_categorical_crossentropy" 
                                else 'trained_bin_model.h5')
        x_data, y_data = load_data_from_csv(f"{output_dir}input_data.csv", f"{output_dir}output_data.csv")
        
        # Load split information
        split_info = np.load(os.path.join(output_dir, 'split_info.npy'), allow_pickle=True).item()
        train_size = split_info['train_size']
        batch_size = split_info['batch_size']
        n_pre = split_info['n_pre']
        num_features = split_info['num_features']
        
        # Create test dataset from remaining data
        test_data_x = x_data[train_size:].copy()
        test_data_y = y_data[train_size:].copy()
        
        logger.info("\nDataset information:")
        logger.info(f"Total samples: {len(x_data)}")
        logger.info(f"Test samples: {len(test_data_x)}")
        logger.info(f"Feature dimension: {num_features}")
        logger.info(f"Sequence length: {n_pre}")
        
        # Create test dataset
        test_dataset = create_dataset(
            test_data_x,
            test_data_y,
            n_pre,
            batch_size,
            loss_fn,
            J
        )
        
        # Calculate test steps accounting for sequence length
        test_steps = max(1, (len(test_data_x) - n_pre + 1) // batch_size)
        logger.info(f"Test steps: {test_steps}")
        
        # Load and verify model
        logger.info("\nLoading model...")
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
            steps=None,  # Let TF handle steps with batched dataset
            verbose=1
        )
        
        logger.info("\nTest metrics:")
        for name, value in zip(model.metrics_names, test_metrics):
            logger.info(f"{name}: {value:.4f}")
        
        logger.info("\nGenerating predictions...")
        preds_test = model.predict(
            test_dataset,
            steps=None,  # Let TF handle steps with batched dataset
            verbose=1
        )
        
        # Ensure we only keep predictions for actual samples
        n_valid_samples = len(test_data_x) - n_pre + 1
        preds_test = preds_test[:n_valid_samples]
        
        logger.info(f"\nPredictions summary:")
        logger.info(f"Shape: {preds_test.shape}")
        logger.info(f"Data type: {preds_test.dtype}")
        logger.info(f"Range: [{np.min(preds_test)}, {np.max(preds_test)}]")
        
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