import os
import sys
import numpy as np
import pandas as pd
from keras.models import load_model

def test_model():
    # Load the saved model
    model = load_model(model_path)

    # Load new input data

    n_post = int(1)
    n_pre = int(window_size)
    seq_len = int(t_end)

    x = np.array(pd.read_csv("{}input_data.csv".format(output_dir)))

    print('raw x shape', x.shape)   

    dX = []
    for i in range(seq_len-n_pre-n_post):
        dX.append(x[i:i+n_pre])
    
    dataX = np.array(dX)

    print('dataX shape:', dataX.shape)

    nb_features = dataX.shape[2]

    print('nb_features:', nb_features)

    # now test

    print('Generate predictions')

    preds_test = model.predict(dataX, batch_size=int(nb_batches), verbose=0)

    print('predictions shape =', preds_test.shape)

    # Save predictions

    print('Saving to {}lstm_new_preds.npy'.format(output_dir))

    np.save("{}lstm_new_preds.npy".format(output_dir), preds_test)


def main():
    test_model()
    return 1

if __name__ == "__main__":
    main()