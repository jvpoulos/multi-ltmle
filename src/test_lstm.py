import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

def test_model():
    if loss_fn == "sparse_categorical_crossentropy":
        model_path = os.path.join(output_dir, 'trained_cat_model.h5')
        input_data = pd.read_csv("{}input_cat_data.csv".format(output_dir), low_memory=False)
        time_step_col = input_data.columns[0]  # Assuming the first column is the time step
        feature_cols = input_data.columns[1:]  # Assuming the remaining columns are features
        x = input_data.pivot(index=time_step_col, columns=feature_cols, values='value').values
    else:
        model_path = os.path.join(output_dir, 'trained_bin_model.h5')
        input_data = pd.read_csv("{}input_bin_data.csv".format(output_dir), low_memory=False)
        time_step_col = input_data.columns[0]  # Assuming the first column is the time step
        feature_cols = input_data.columns[1:]  # Assuming the remaining columns are features
        x = input_data.pivot(index=time_step_col, columns=feature_cols, values='value').values

    model = load_model(model_path)

    n_pre = int(window_size)
    seq_len = int(t_end)

    dX = []
    for i in range(seq_len - n_pre):
        dX.append(x[i:i+n_pre])

    dataX = np.array(dX)
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