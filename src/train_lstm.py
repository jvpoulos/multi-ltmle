import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

from utils import prepare_datasets, data_generator, create_model, load_data_from_csv

import sys
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_excepthook(type, value, tb):
    logger.error("An error occurred:")
    logger.error(f"Type: {type}")
    logger.error(f"Value: {value}")
    logger.error("Traceback:")
    traceback.print_tb(tb)

sys.excepthook = custom_excepthook

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def main():
    global n_pre, nb_batches, output_dir, loss_fn, epochs, lr, dr, n_hidden, hidden_activation, out_activation, patience, J, window_size

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

    try:
        x_data, y_data = load_data_from_csv(f"{output_dir}input_data.csv", f"{output_dir}output_data.csv")

        logger.info("Data loaded successfully")
        logger.info(f"x_data shape: {x_data.shape}")
        logger.info(f"y_data shape: {y_data.shape}")

        # Ensure y_data contains the 'A' columns
        y_columns = [col for col in y_data.columns if col.startswith('A')]
        if not y_columns:
            logger.warning("No 'A' columns found in y_data. Creating dummy 'A' columns.")
            for i in range(J):
                y_data[f'A{i}'] = 0
            y_columns = [f'A{i}' for i in range(J)]

        # Ensure 'ID' column exists in both x_data and y_data
        if 'ID' not in x_data.columns:
            x_data['ID'] = range(len(x_data))
        if 'ID' not in y_data.columns:
            y_data['ID'] = range(len(y_data))

        # Keep 'ID' column in y_data
        y_data = y_data[['ID'] + y_columns]

        # Convert y_data to integer type
        y_data[y_columns] = y_data[y_columns].astype(int)

        logger.info("Data shapes:")
        logger.info(f"x shape: {x_data.shape}")
        logger.info(f"y shape: {y_data.shape}")

        logger.info("x_data head:")
        logger.info(x_data.head())
        logger.info("y_data head:")
        logger.info(y_data.head())

    except Exception as e:
        logger.error(f"Error during data loading and processing: {str(e)}")
        logger.error(traceback.format_exc())
        return

    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)
    
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Train size: {train_size}")
    logger.info(f"Batch size: {batch_size}")

    # Adjust batch_size if it's larger than the number of samples
    batch_size = min(batch_size, num_samples)

    # Use prepare_datasets function
    train_dataset, val_dataset, train_size, val_size = prepare_datasets(x_data, y_data[y_columns], n_pre, batch_size, validation_split=0.2, loss_fn=loss_fn)

    steps_per_epoch = max(1, train_size // batch_size)
    validation_steps = max(1, val_size // batch_size)

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")
    logger.info(f"Train dataset size: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    logger.info(f"Validation dataset size: {tf.data.experimental.cardinality(val_dataset).numpy()}")

    for x_batch, y_batch in train_dataset.take(1):
        logger.info(f"X batch shape: {x_batch.shape}")
        logger.info(f"Y batch shape: {y_batch.shape}")
        
    input_shape = (n_pre, x_data.shape[1] - 1)  # Subtract 1 to exclude the ID column
    logger.info(f"Input shape: {input_shape}")
    output_dim = J  # Use J instead of 1 for sparse_categorical_crossentropy

    logger.info(f"Creating model with input_shape={input_shape}, output_dim={output_dim}")
    model = create_model(input_shape, output_dim, lr, dr, n_hidden, hidden_activation, out_activation, loss_fn, J)

    logger.info("Model summary:")
    model.summary()

    # Define callbacks
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_dir, 'model_{epoch}.h5'),
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=int(patience),
        min_delta=0,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    terminate_on_nan = TerminateOnNaN()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    try:
        history = model.fit(
            train_dataset.repeat(),
            epochs=int(epochs),
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset.repeat(),
            validation_steps=validation_steps,
            callbacks=[early_stopping, terminate_on_nan, checkpoint, reduce_lr],
            verbose=1
        )
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return

    # Ensure the directory exists before saving
    os.makedirs(output_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(output_dir, 'trained_cat_model.h5' if loss_fn == "sparse_categorical_crossentropy" else 'trained_bin_model.h5')
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Make predictions on the entire dataset
    all_data = tf.data.Dataset.from_generator(
        lambda: data_generator(x_data.drop(columns=['ID']), y_data, n_pre, batch_size, loss_fn),
        output_signature=(
            tf.TensorSpec(shape=(None, n_pre, x_data.shape[1] - 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32 if loss_fn == "sparse_categorical_crossentropy" else tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    try:
        predictions = []
        for batch in all_data:
            batch_predictions = model.predict_on_batch(batch[0])
            predictions.append(batch_predictions)
        predictions = np.concatenate(predictions, axis=0)
        logger.info(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        logger.error(f"An error occurred during prediction: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return

    # Save predictions
    pred_path = os.path.join(output_dir, 'lstm_cat_preds.npy' if loss_fn == "sparse_categorical_crossentropy" else 'lstm_bin_preds.npy')
    np.save(pred_path, predictions)
    logger.info(f"Predictions saved to: {pred_path}")

    # Save detailed information
    info_path = os.path.join(output_dir, 'lstm_cat_preds_info.npz' if loss_fn == "sparse_categorical_crossentropy" else 'lstm_bin_preds_info.npz')
    np.savez(info_path, 
             shape=predictions.shape,
             dtype=str(predictions.dtype),
             min_value=np.min(predictions),
             max_value=np.max(predictions),
             num_samples=num_samples)
    
if __name__ == "__main__":
    main()