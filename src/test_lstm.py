import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

def test_model():
    if loss_fn == "sparse_categorical_crossentropy":
        model_path = os.path.join(output_dir, 'trained_cat_model.h5')
        x = np.array(pd.read_csv("{}input_cat_data.csv".format(output_dir)))
    else:
        model_path = os.path.join(output_dir, 'trained_bin_model.h5')
        x = np.array(pd.read_csv("{}input_bin_data.csv".format(output_dir)))

    model = load_model(model_path)

    n_post = int(1)
    n_pre = int(window_size)
    seq_len = int(t_end)

    dX = []
    for i in range(seq_len - n_pre - n_post):
        dX.append(x[i:i + n_pre])

    dataX = np.array(dX)

    print('dataX shape:', dataX.shape)
    nb_features = dataX.shape[2]  # Get the number of features from the third dimension
    print('nb_features:', nb_features)

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