import os
import math
import numpy as np
import pandas as pd
import time
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

import wandb
from datetime import datetime

class FilterPatterns(logging.Filter):
    def __init__(self, patterns):
        super().__init__()
        self.patterns = patterns

    def filter(self, record):
        # Return False to filter out, True to keep
        return not any(pattern in record.getMessage() for pattern in self.patterns)

def log_metrics(history, start_time):
    metrics_to_log = {}
    
    # Safely get metrics with fallbacks
    try:
        metrics_to_log.update({
            'final_train_loss': history.history.get('loss', [0])[-1],
            'final_val_loss': history.history.get('val_loss', [0])[-1],
            'final_train_accuracy': history.history.get('accuracy', [0])[-1],
            'final_val_accuracy': history.history.get('val_accuracy', [0])[-1],
            'best_val_loss': min(history.history.get('val_loss', [float('inf')])),
            'best_val_accuracy': max(history.history.get('val_accuracy', [0])),
            'training_time': time.time() - start_time
        })
        
        # Add sparse categorical accuracy metrics if they exist
        if 'sparse_categorical_accuracy' in history.history:
            metrics_to_log.update({
                'final_train_sparse_categorical_accuracy': history.history['sparse_categorical_accuracy'][-1],
                'final_val_sparse_categorical_accuracy': history.history['val_sparse_categorical_accuracy'][-1]
            })
            
    except Exception as e:
        logger.warning(f"Error collecting metrics: {str(e)}")
        metrics_to_log.update({
            'error_collecting_metrics': str(e)
        })
    
    wandb.log(metrics_to_log)

def setup_wandb(config, validation_steps=None, train_dataset=None):
    """Initialize WandB with configuration.
    
    Args:
        config (dict): WandB configuration dictionary
        validation_steps (int, optional): Number of validation steps
        train_dataset: Training dataset for batch logging
    """
    run = wandb.init(
        project="multi-ltmle",
        entity="jvpoulos",
        config=config,
        name=f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Configure WandB callback with SavedModel format
    callback_config = {
        'monitor': 'val_loss',
        'log_weights': True,
        'log_gradients': True,
        'log_evaluation': True,
        'save_model': False  # Disable default HDF5 saving
    }
    
    # Add optional configurations if provided
    if validation_steps is not None:
        callback_config['validation_steps'] = validation_steps
    
    if train_dataset is not None:
        callback_config['training_data'] = train_dataset
        callback_config['log_batch_frequency'] = 100
    
    wandb_callback = wandb.keras.WandbCallback(**callback_config)
    
    return run, wandb_callback

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_dataset):
        super(CustomCallback, self).__init__()  # Add proper super() call
        self.train_dataset = train_dataset
        self.start_time = time.time()
        self.model = None  # Initialize model attribute
 
    def set_model(self, model):
        """This is required to associate the model with the callback."""
        super(CustomCallback, self).set_model(model)  # Add proper super() call
        self.model = model
               
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}
            
        epoch_time = time.time() - self.epoch_start_time
        
        if self.model is None:
            logger.warning("Model not set in CustomCallback")
            return
            
        # Log metrics to WandB
        wandb.log({
            'epoch': epoch,
            'epoch_time': epoch_time,
            'learning_rate': K.eval(self.model.optimizer.learning_rate),
            **logs
        })
        
        try:
            # Sample predictions
            for x_batch, y_batch in self.train_dataset.take(1):
                sample_pred = self.model.predict(x_batch)
                
                # Log prediction statistics
                wandb.log({
                    'sample_predictions_mean': np.mean(sample_pred),
                    'sample_predictions_std': np.std(sample_pred),
                    'sample_predictions_hist': wandb.Histogram(sample_pred)
                })
                break  # Only take one batch
        except Exception as e:
            logger.warning(f"Error in CustomCallback prediction logging: {str(e)}")
    
    def on_train_batch_end(self, batch, logs=None):
        if not logs:
            logs = {}
            
        # Log batch-level metrics periodically
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
            mode='min'
        ),
        
        # Best model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        
        # Regular checkpoints (keep last 3)
        CustomModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.h5'),
            keep_n=3,
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

def get_output_signature(loss_fn, J):
    if loss_fn == "sparse_categorical_crossentropy":
        return tf.TensorSpec(shape=(None,), dtype=tf.int32)
    elif loss_fn == "binary_crossentropy":
        return tf.TensorSpec(shape=(None,), dtype=tf.float32)
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

# In utils.py, modify the load_data_from_csv function:

def load_data_from_csv(input_file, output_file):
    """Load and preprocess data from CSV files."""
    try:
        # Load input data
        x_data = pd.read_csv(input_file)
        logger.info(f"Successfully loaded input data from {input_file}")
        logger.info(f"Input data shape: {x_data.shape}")
        
        # Load output data
        y_data = pd.read_csv(output_file)
        logger.info(f"Successfully loaded output data from {output_file}")
        logger.info(f"Output data shape: {y_data.shape}")
        
        # Print raw data info for debugging
        print("Raw y_data columns:", y_data.columns)
        print("Raw y_data unique values:", {col: y_data[col].unique() for col in y_data.columns})
        
        # Ensure 'ID' column exists in x_data
        if 'ID' not in x_data.columns:
            if 'id' in x_data.columns:
                x_data['ID'] = x_data['id'].astype(int)
                x_data = x_data.drop('id', axis=1)
            else:
                x_data['ID'] = range(len(x_data))
                logger.info("Added sequential ID column to x_data")
        
        # Process y_data
        if y_data.empty:
            logger.warning("y_data is empty, creating synthetic treatment assignments")
            y_data = pd.DataFrame({'ID': x_data['ID']})
            for i in range(6):  # J=6 treatments
                y_data[f'A{i}'] = np.random.choice([0, 1], size=len(y_data), p=[0.5, 0.5])
        else:
            # Handle ID column in y_data
            if 'ID' not in y_data.columns:
                if 'id' in y_data.columns:
                    y_data['ID'] = y_data['id'].astype(int)
                    y_data = y_data.drop('id', axis=1)
                else:
                    y_data['ID'] = range(len(y_data))
            
            # Find treatment columns using both patterns (A. and A)
            A_columns = [col for col in y_data.columns 
                        if col.startswith('A.') or (col.startswith('A') and not col.endswith('_x') and not col.endswith('_y'))]
            
            if not A_columns:
                logger.warning("No treatment columns found in y_data. Creating dummy 'A' columns.")
                for i in range(6):  # J=6 treatments
                    y_data[f'A{i}'] = np.random.choice([0, 1], size=len(y_data))
                A_columns = [f'A{i}' for i in range(6)]
            
            # Clean column names - remove dots and clean up
            y_data.columns = [col.replace('.', '') for col in y_data.columns]
            A_columns = [col.replace('.', '') for col in A_columns]
                            
            # Keep only ID and treatment columns
            y_data = y_data[['ID'] + A_columns]
        
        # Merge data
        merged_data = pd.merge(x_data, y_data, on='ID', how='inner', suffixes=('', '_y'))
        logger.info(f"Merged data shape: {merged_data.shape}")
        
        # Get clean column lists
        x_columns = [col for col in x_data.columns if col != 'ID']
        y_columns = [col for col in y_data.columns if col != 'ID']
        
        # Split back into x_data and y_data
        x_data = merged_data[['ID'] + x_columns]
        y_data = merged_data[y_columns]
        
        # Fill NaN values and convert types
        x_data = x_data.fillna(-1).astype(np.float32)
        y_data = y_data.fillna(0).astype(np.float32)
        
        # Print verification info
        logger.info(f"Final x_data shape: {x_data.shape}")
        logger.info(f"Final y_data shape: {y_data.shape}")
        logger.info(f"Treatment columns: {y_columns}")
        logger.info(f"Final y_data unique values: {[y_data[col].unique() for col in y_columns]}")
        
        if y_data.empty or y_data.shape[1] == 0:
            raise ValueError("No treatment columns in final y_data")
            
        return x_data, y_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Input file exists: {os.path.exists(input_file)}")
        logger.error(f"Output file exists: {os.path.exists(output_file)}")
        raise

def configure_gpu(policy=None):
    """Configure GPU to avoid XLA/update issues and handle version mismatches."""
    try:
        # Reset any existing GPU configurations
        tf.keras.backend.clear_session()
        K.clear_session()
        gc.collect()
        
        # Enhanced logging configuration
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        # Disable AutoGraph warnings
        tf.autograph.set_verbosity(0)
        tf.debugging.disable_traceback_filtering()
        
        # Essential CUDA/cuDNN configurations
        os.environ.update({
            'CUDA_CACHE_DISABLE': '1',
            'TF_CUDNN_USE_AUTOTUNE': '0',
            'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
            'TF_CUDNN_DETERMINISTIC': '1',
            'TF_ENABLE_CUDNN_FRONTEND_FALLBACK': '1',
            'TF_ENABLE_CUDNN_RNN_BACKEND': '1',
            'TF_GPU_THREAD_MODE': 'gpu_private',
            'TF_GPU_THREAD_COUNT': '1',
            'TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT': '0',
            'TF_ENABLE_AUTO_MIXED_PRECISION': '0',
            'TF_SYNC_ON_FINISH': '0',
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
            'TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE': 'False'
        })
        
        # Completely disable XLA and JIT
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0 --tf_xla_async_compilation=false'
        tf.config.optimizer.set_jit(False)
        
        # Force graph mode execution
        tf.config.run_functions_eagerly(False)
        tf.compat.v1.disable_eager_execution_outside_functions()
        
        # Get available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception as e:
                        logger.warning(f"Could not set memory growth for {gpu.name}: {str(e)}")
                
                # Configure visible devices based on environment
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    try:
                        visible_gpus = [gpus[int(i)] for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if i.isdigit()]
                        if visible_gpus:
                            tf.config.set_visible_devices(visible_gpus, 'GPU')
                    except Exception as e:
                        logger.warning(f"Error setting visible devices: {str(e)}")
                        tf.config.set_visible_devices(gpus, 'GPU')
                else:
                    tf.config.set_visible_devices(gpus, 'GPU')
                
                # Configure memory limits
                for gpu in gpus:
                    try:
                        import subprocess
                        gpu_id = gpu.name.split(':')[-1].strip()
                        result = subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits', '--id=' + gpu_id],
                            stderr=subprocess.DEVNULL
                        )
                        total_memory = int(result.decode().strip())
                        memory_limit = int(total_memory * 0.9)
                        
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                        )
                    except Exception:
                        # Fallback to conservative memory limit
                        try:
                            tf.config.set_logical_device_configuration(
                                gpu,
                                [tf.config.LogicalDeviceConfiguration(memory_limit=6144)]
                            )
                        except Exception as e:
                            logger.warning(f"Could not set memory limit for {gpu.name}: {str(e)}")
                
                # Force FP32 precision
                tf.keras.mixed_precision.set_global_policy('float32')
                
                # Configure distribution strategy options
                tf.distribute.InputOptions(
                    experimental_fetch_to_device=False,
                    experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA
                )
                
                # Disable auto-graph 
                tf.config.experimental.disable_mlir_graph_optimization()
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)
                
                # Log successful configuration
                logger.info(f"Found {len(gpus)} GPU(s)")
                for gpu in gpus:
                    logger.info(f"Configured GPU: {gpu.name}")
                return True
                
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {str(e)}")
                return False
        else:
            logger.warning("No GPU found, using CPU")
            return False
            
    except Exception as e:
        logger.warning(f"GPU configuration failed, using CPU: {str(e)}")
        return False

def get_strategy():
    """Get appropriate distribution strategy based on available GPUs."""
    if not hasattr(get_strategy, "strategy"):
        try:
            # Disable XLA
            tf.config.optimizer.set_jit(False)
            
            # Force graph execution
            tf.config.run_functions_eagerly(False)
            
            # Use default strategy with synchronous updates
            get_strategy.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            
            # Configure distribution
            options = tf.distribute.InputOptions(
                experimental_fetch_to_device=False,
                experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA
            )
            get_strategy.strategy.extended.experimental_enable_get_next_as_optional = False
            
            logger.info("Using OneDeviceStrategy with synchronous updates")
        except:
            get_strategy.strategy = tf.distribute.get_strategy()
            logger.info("Using default strategy")
    
    return get_strategy.strategy

def create_dataset(x_data, y_data, n_pre, batch_size, loss_fn, J, is_training=False):
    """Create dataset with proper sequence handling."""
    # Remove ID column if present
    if 'ID' in x_data.columns:
        x_data = x_data.drop('ID', axis=1)
    
    num_features = x_data.shape[1]
    n_samples = max(0, len(x_data) - n_pre + 1)
    
    logger.info(f"Creating sequences from {len(x_data)} samples with {n_pre} timesteps")
    logger.info(f"Input shape: {x_data.shape}, Output shape: {y_data.shape}")
    
    def prepare_sequences():
        # Convert to numpy and use float32 for memory efficiency
        x_values = x_data.values.astype(np.float32)
        y_values = y_data.values.astype(np.float32)
        
        # Calculate normalization statistics
        mean = np.nanmean(x_values, axis=0)
        std = np.nanstd(x_values, axis=0)
        std[std == 0] = 1
        
        sequences_x = []
        sequences_y = []
        
        for i in range(n_samples):
            try:
                # Create input sequence
                x_seq = x_values[i:i + n_pre].copy()
                x_seq = (x_seq - mean) / std
                
                if is_training:
                    # Add small random noise for regularization
                    x_seq += np.random.normal(0, 0.001, x_seq.shape)
                
                # Get target values for this sequence
                y_target = y_values[i + n_pre - 1]
                
                # Handle target based on loss function
                if loss_fn == "sparse_categorical_crossentropy":
                    # Find the position of the maximum value in the target vector
                    y_seq = np.argmax(y_target)
                    # Ensure it's within bounds
                    y_seq = y_seq % J if y_seq >= J else y_seq
                else:
                    # For binary classification, check if any value is above threshold
                    y_seq = float(np.any(y_target > 0.5))
                
                # Store sequences
                sequences_x.append(x_seq.astype(np.float32))
                sequences_y.append(np.array(y_seq, 
                                         dtype=np.int32 if loss_fn == "sparse_categorical_crossentropy" 
                                         else np.float32))
                
            except Exception as e:
                logger.warning(f"Error processing sequence {i}: {str(e)}")
                continue
        
        if not sequences_x:
            raise ValueError("No valid sequences created")
        
        # Convert to numpy arrays
        sequences_x = np.array(sequences_x, dtype=np.float32)
        sequences_y = np.array(sequences_y, dtype=np.int32 if loss_fn == "sparse_categorical_crossentropy" else np.float32)
        
        # Print info about sequences
        logger.info(f"Created {len(sequences_x)} sequences")
        logger.info(f"X shape: {sequences_x.shape}, Y shape: {sequences_y.shape}")
        logger.info(f"Y unique values: {np.unique(sequences_y)}")
        
        return sequences_x, sequences_y
    
    try:
        # Create sequences
        x_sequences, y_sequences = prepare_sequences()
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((x_sequences, y_sequences))
        
        # Configure dataset options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        options.deterministic = False
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        
        # Apply options and augmentation
        dataset = dataset.with_options(options)
        
        if is_training:
            dataset = dataset.cache()
            dataset = dataset.shuffle(min(5000, n_samples), reshuffle_each_iteration=True)
            
            # Add simple data augmentation for time series
            def augment(x, y):
                # Add small random noise
                noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.001)
                x = x + noise
                return x, y
            
            dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset, n_samples
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        logger.error("Input data info:")
        logger.error(f"x_data shape: {x_data.shape}")
        logger.error(f"y_data shape: {y_data.shape}")
        logger.error(f"y_data unique values: {[y_data[col].unique() for col in y_data.columns]}")
        raise

def create_model(input_shape, output_dim, lr, dr, n_hidden, hidden_activation, out_activation, loss_fn, J, epochs, steps_per_epoch, strategy=None):
    """Create model with simplified architecture to avoid XLA/GPU issues."""
    if strategy is None:
        strategy = get_strategy()
    
    with strategy.scope():
        # Configure model for graph mode
        tf.config.run_functions_eagerly(False)
        
        # Input layer
        inputs = Input(shape=input_shape, dtype='float32')
        
        # Simple RNN config without advanced features
        rnn_config = {
            'activation': 'tanh',  # Back to tanh for simplicity
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': l2(0.0001),
            'dtype': 'float32',
            'recurrent_initializer': 'orthogonal'
        }
        
        # First RNN block
        x = tf.keras.layers.SimpleRNN(
            n_hidden * 2,
            return_sequences=True,
            **rnn_config
        )(inputs)
        x = tf.keras.layers.Dropout(dr)(x)
        
        # Second RNN block
        x = tf.keras.layers.SimpleRNN(
            n_hidden,
            return_sequences=True,
            **rnn_config
        )(x)
        x = tf.keras.layers.Dropout(dr)(x)
        
        # Third RNN block
        x = tf.keras.layers.SimpleRNN(
            max(32, n_hidden // 2),
            return_sequences=False,
            **rnn_config
        )(x)
        x = tf.keras.layers.Dropout(dr)(x)
        
        # Dense layer
        x = tf.keras.layers.Dense(
            n_hidden,
            activation='relu',
            kernel_regularizer=l2(0.0001)
        )(x)
        x = tf.keras.layers.Dropout(dr)(x)
        
        if loss_fn == "sparse_categorical_crossentropy":
            outputs = tf.keras.layers.Dense(
                J,
                activation='softmax'
            )(x)
            
            def custom_loss(y_true, y_pred):
                y_true = tf.cast(y_true, tf.int32)
                y_pred = tf.cast(y_pred, tf.float32)
                
                scce = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False,
                    reduction=tf.keras.losses.Reduction.NONE
                )
                base_loss = scce(y_true, y_pred)
                
                class_weights = tf.constant([0.1] + [1.0] * (J-1), dtype=tf.float32)
                weights = tf.gather(class_weights, tf.cast(y_true, tf.int32))
                
                return tf.reduce_mean(base_loss * weights)
            
            loss = custom_loss
        else:
            outputs = tf.keras.layers.Dense(
                1,
                activation='sigmoid'
            )(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs, outputs)
        
        # Basic optimizer configuration
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr
        )
        
        metrics = ['accuracy']
        if loss_fn == "sparse_categorical_crossentropy":
            metrics.append(tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy'))

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            run_eagerly=False,
            jit_compile=False
        )
        
        return model