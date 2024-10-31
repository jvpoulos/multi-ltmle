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
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

from utils import data_generator, create_model, load_data_from_csv, get_output_signature, create_dataset, get_training_config, get_optimized_callbacks

import sys
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                y_data[f'A{i}'] = np.random.choice([0, 1], size=len(y_data))  # Random binary values
            y_columns = [f'A{i}' for i in range(J)]

        # Ensure 'ID' column exists in both x_data and y_data
        if 'ID' not in x_data.columns:
            x_data['ID'] = range(len(x_data))
        if 'ID' not in y_data.columns:
            y_data['ID'] = range(len(y_data))

        # Keep 'ID' column and 'A' columns in y_data
        y_data = y_data[['ID'] + y_columns]

        # Convert y_data to integer type
        y_data[y_columns] = y_data[y_columns].astype(int)

        logger.info("Data shapes after processing:")
        logger.info(f"x shape: {x_data.shape}")
        logger.info(f"y shape: {y_data.shape}")

        logger.info("x_data head:")
        logger.info(x_data.head())
        logger.info("y_data head:")
        logger.info(y_data.head())

        if y_data.empty:
            raise ValueError("y_data is still empty after processing. Cannot proceed with training.")

    except Exception as e:
        logger.error(f"Error during data loading and processing: {str(e)}")
        logger.error(traceback.format_exc())
        return

    # After loading and processing data, add:
    num_samples = len(x_data)
    train_size = int(0.8 * num_samples)  # 80% for training/validation
    val_size = int(0.2 * train_size)    # 20% of training data for validation
    final_train_size = train_size - val_size

    # Calculate feature dimension
    num_features = len(x_data.columns)
    if 'ID' in x_data.columns:
        num_features -= 1

    logger.info(f"Dataset splits:")
    logger.info(f"Total samples: {num_samples}")
    logger.info(f"Training samples: {final_train_size}")
    logger.info(f"Validation samples: {val_size}")
    logger.info(f"Test samples: {num_samples - train_size}")
    logger.info(f"Number of features: {num_features}")
    logger.info(f"Feature columns: {x_data.columns.tolist()}")

    # Create training dataset
    train_dataset = create_dataset(
        x_data[:final_train_size].copy(), 
        y_data[:final_train_size].copy(), 
        n_pre, 
        batch_size, 
        loss_fn, 
        J
    )

    # Create validation dataset
    val_dataset = create_dataset(
        x_data[final_train_size:train_size].copy(), 
        y_data[final_train_size:train_size].copy(), 
        n_pre, 
        batch_size, 
        loss_fn, 
        J
    )

    # Calculate steps
    steps_per_epoch = max(1, (final_train_size - n_pre + 1) // batch_size)
    validation_steps = max(1, (val_size - n_pre + 1) // batch_size)

    logger.info(f"Training steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")

    # Verify dataset shapes
    for dataset_name, dataset in [("Train", train_dataset), ("Validation", val_dataset)]:
        for x_batch, y_batch in dataset.take(1):
            logger.info(f"\n{dataset_name} dataset characteristics:")
            logger.info(f"X shape: {x_batch.shape}")
            logger.info(f"Y shape: {y_batch.shape}")
            logger.info(f"Feature dimension: {x_batch.shape[-1]}")
            logger.info(f"Expected features: {num_features}")
            if x_batch.shape[-1] != num_features:
                raise ValueError(
                    f"Feature dimension mismatch in {dataset_name} dataset. "
                    f"Expected {num_features}, got {x_batch.shape[-1]}"
                )
    
    # Save split information for test_lstm.py
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
    
    input_shape = (n_pre, num_features)
    logger.info(f"Input shape: {input_shape}")
    if loss_fn == "sparse_categorical_crossentropy":
        output_dim = J
    elif loss_fn == "binary_crossentropy":
        output_dim = 1
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info(f"GPU memory growth enabled on {len(physical_devices)} devices")
        except RuntimeError as e:
            logger.warning(f"Error setting memory growth: {e}")

    logger.info(f"Creating model with input_shape={input_shape}, output_dim={output_dim}")
    # Update distribution strategy setup
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        # Create model
        model = create_model(input_shape, output_dim, lr, dr, n_hidden, 
                            hidden_activation, out_activation, loss_fn, J)


    # Get optimized training configuration
    train_config = get_training_config(epochs, batch_size, steps_per_epoch, validation_steps)
    
    # Get optimized callbacks with train_dataset
    callbacks = get_optimized_callbacks(patience, output_dir, train_dataset)

    try:
        # Train with repeat after cache
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate the model on validation set
        logger.info("\nEvaluating validation set...")
        val_metrics = model.evaluate(val_dataset, steps=validation_steps)
        
        # Log all metrics
        logger.info("\nValidation metrics:")
        for name, value in zip(model.metrics_names, val_metrics):
            logger.info(f"{name}: {value:.4f}")
        
        # Store validation metrics separately if needed
        val_loss = val_metrics[0]  # Loss is always first
        val_accuracy = val_metrics[1] if len(val_metrics) > 1 else None  # Accuracy might be second
        
        logger.info(f"\nValidation loss: {val_loss:.4f}")
        if val_accuracy is not None:
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")

        # Make some predictions
        logger.info("\nGenerating sample predictions...")
        for x_batch, y_batch in val_dataset.take(1):
            predictions = model.predict(x_batch)
            logger.info(f"Sample predictions: {predictions[:10]}")
            logger.info(f"Sample true labels: {y_batch[:10]}")
            
            # Print prediction statistics
            logger.info("\nPrediction statistics:")
            logger.info(f"Mean prediction: {np.mean(predictions):.4f}")
            logger.info(f"Std prediction: {np.std(predictions):.4f}")
            logger.info(f"Min prediction: {np.min(predictions):.4f}")
            logger.info(f"Max prediction: {np.max(predictions):.4f}")
            
            # Class distribution
            if loss_fn == "binary_crossentropy":
                pred_classes = (predictions > 0.5).astype(int)
                logger.info(f"\nPredicted class distribution: {np.bincount(pred_classes.flatten())}")
                logger.info(f"True class distribution: {np.bincount(y_batch.numpy().astype(int))}")
            else:
                pred_classes = np.argmax(predictions, axis=1)
                logger.info(f"\nPredicted class distribution: {np.bincount(pred_classes)}")
                logger.info(f"True class distribution: {np.bincount(y_batch.numpy())}")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(traceback.format_exc())
        return

    # Ensure the directory exists before saving
    os.makedirs(output_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(output_dir, 'trained_cat_model.h5' if loss_fn == "sparse_categorical_crossentropy" else 'trained_bin_model.h5')
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Create data for final predictions (after model saving)
    try:
        logger.info("\nGenerating predictions for all data...")
        
        # Prepare data without ID column
        x_data_final = x_data.copy()
        if 'ID' in x_data_final.columns:
            x_data_final = x_data_final.drop(columns=['ID'])
        elif 'id' in x_data_final.columns:
            x_data_final = x_data_final.drop(columns=['id'])
            
        # Create dataset for predictions
        all_data = create_dataset(
            x_data_final,
            y_data,
            n_pre,
            batch_size,
            loss_fn,
            J
        )
        
        # Calculate steps for full dataset
        total_steps = max(1, (len(x_data_final) - n_pre + 1) // batch_size)
        logger.info(f"Total prediction steps: {total_steps}")
        
        # Generate predictions
        predictions = []
        for batch in all_data:
            batch_predictions = model.predict_on_batch(batch[0])
            predictions.append(batch_predictions)
        
        # Concatenate all predictions
        predictions = np.concatenate(predictions, axis=0)
        
        # Ensure we only keep valid predictions
        n_valid_predictions = len(x_data_final) - n_pre + 1
        predictions = predictions[:n_valid_predictions]
        
        logger.info(f"Predictions shape: {predictions.shape}")
        
        # Save predictions
        pred_path = os.path.join(output_dir, 
                               'lstm_cat_preds.npy' if loss_fn == "sparse_categorical_crossentropy" 
                               else 'lstm_bin_preds.npy')
        np.save(pred_path, predictions)
        logger.info(f"Predictions saved to: {pred_path}")
        
        # Save detailed information
        info_path = os.path.join(output_dir, 
                               'lstm_cat_preds_info.npz' if loss_fn == "sparse_categorical_crossentropy" 
                               else 'lstm_bin_preds_info.npz')
        np.savez(
            info_path,
            shape=predictions.shape,
            dtype=str(predictions.dtype),
            min_value=np.min(predictions),
            max_value=np.max(predictions),
            num_samples=n_valid_predictions,
            num_features=num_features
        )
        logger.info(f"Detailed information saved to: {info_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during prediction or saving: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
if __name__ == "__main__":
    main()