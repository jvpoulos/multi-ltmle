import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

def test_model():
    if loss_fn == "sparse_categorical_crossentropy":
        model_path = os.path.join(output_dir, 'trained_cat_model.h5')
        input_data = pd.read_csv("{}input_cat_data.csv".format(output_dir), low_memory=False)
        time_step_col = 't'
        id_col = 'id'
        feature_cols = 'feature'
        x = input_data.pivot_table(index=[id_col, time_step_col], columns=feature_cols, values='value', aggfunc='first').values
    else:
        model_path = os.path.join(output_dir, 'trained_bin_model.h5')
        input_data = pd.read_csv("{}input_bin_data.csv".format(output_dir), low_memory=False)
        time_step_col = 't'
        id_col = 'id'
        feature_cols = 'feature'
        x = input_data.pivot_table(index=[id_col, time_step_col], columns=feature_cols, values='value', aggfunc='first').values
    
    model = load_model(model_path)
    
    n_pre = int(window_size)
    
    num_individuals = x.shape[0]
    num_timesteps = x.shape[1]
    dataX = np.zeros((num_individuals, num_timesteps - n_pre + 1, n_pre, x.shape[2]))
    for i in range(num_individuals):
        for j in range(num_timesteps - n_pre + 1):
            dataX[i, j, :, :] = x[i, j:j+n_pre, :]
        
    print('dataX shape:', dataX.shape)
    
    print('Generate predictions')
    preds_test = model.predict(dataX, batch_size=int(nb_batches), verbose=1)
    
    if loss_fn == "sparse_categorical_crossentropy":
        print('Saving predictions to {}lstm_new_cat_preds.npy'.format(output_dir))
        np.save("{}lstm_new_cat_preds.npy".format(output_dir), preds_test)
    else:
        print('Saving predictions to {}lstm_new_bin_preds.npy'.format(output_dir))
        np.save("{}lstm_new_bin_preds.npy".format(output_dir), preds_test)

def main():
    test_model()

if __name__ == "__main__":
    main()