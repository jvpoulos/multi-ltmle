import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from utils import prepare_datasets, data_generator, load_data_from_csv

import sys
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def test_model():
    global window_size, output_dir, nb_batches, loss_fn, J

    model_path = os.path.join(output_dir, 'trained_cat_model.h5' if loss_fn == "sparse_categorical_crossentropy" else 'trained_bin_model.h5')
    x_data, y_data = load_data_from_csv(f"{output_dir}input_data.csv", f"{output_dir}output_data.csv")
    
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.int32)

    n_pre = int(window_size)
    batch_size = int(nb_batches)

    logger.info("Data shapes:")
    logger.info(f"x shape: {x_data.shape}")
    logger.info(f"y shape: {y_data.shape}")

    logger.info(x_data.head())
    logger.info(y_data.head())

    num_samples = len(x_data)

    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Batch size: {batch_size}")

    # Adjust batch_size if it's larger than the number of samples
    batch_size = min(batch_size, num_samples)

    # Use the 'ID' column if it exists, otherwise use the first column
    id_column = 'ID' if 'ID' in x_data.columns else x_data.columns[0]
    
    # Use prepare_datasets function
    test_dataset, _, test_size, _ = prepare_datasets(x_data.drop(columns=[id_column]), y_data, n_pre, batch_size, validation_split=0, loss_fn=loss_fn)

    steps = max(1, test_size // batch_size)
    logger.info(f"Steps: {steps}")

    model = load_model(model_path)

    logger.info("Model summary:")
    model.summary()

    # Debug: Check the shape of the first batch
    for x_batch, y_batch in test_dataset.take(1):
        logger.info(f"X batch shape: {x_batch.shape}")
        logger.info(f"Y batch shape: {y_batch.shape}")

    try:
        preds_test = model.predict(test_dataset, steps=steps, verbose=1)
        preds_test = preds_test[:num_samples]  # Slice to get predictions for all unique samples
    except Exception as e:
        logger.info(f"An error occurred during prediction: {str(e)}")
        logger.info(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

    logger.info(f"Predictions shape: {preds_test.shape}")
    logger.info(f"Predictions data type: {preds_test.dtype}")
    logger.info(f"Predictions min value: {np.min(preds_test)}")
    logger.info(f"Predictions max value: {np.max(preds_test)}")

    output_file = f"{output_dir}lstm_new_cat_preds.npy" if loss_fn == "sparse_categorical_crossentropy" else f"{output_dir}lstm_new_bin_preds.npy"
    np.save(output_file, preds_test)
    logger.info(f"Predictions saved to: {output_file}")

    # Save detailed information
    info_path = os.path.join(output_dir, 'lstm_new_cat_preds_info.npz' if loss_fn == "sparse_categorical_crossentropy" else 'lstm_new_bin_preds_info.npz')
    np.savez(info_path, 
             shape=preds_test.shape,
             dtype=str(preds_test.dtype),
             min_value=np.min(preds_test),
             max_value=np.max(preds_test),
             num_samples=num_samples)
    logger.info(f"Detailed information saved to: {info_path}")

def main():
    test_model()

if __name__ == "__main__":
    main()