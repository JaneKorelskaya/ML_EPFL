import numpy as np
from argparse import ArgumentParser
import os

from helpers import load_csv_data, create_csv_submission
from implementations import sigmoid, reg_logistic_regression

np.random.seed(10)


def scaler(x, m=0, std=0, train=True):
    if train:
        m = np.mean(x, axis=0)
        std = np.std(x, axis=0)
    x = (x - m) / std
    return x, m, std


def process_data(x, y, x_test, y_test):
    x = np.delete(x, [5, 6, 12, 26, 27, 28, 29, 25], axis=1)
    x[:,0][x[:,0] == -999] = -5
    x[:,20][x[:,20] == -999] = -5
    
    x_test = np.delete(x_test, [5, 6, 12, 26, 27, 28, 29, 25], axis=1)
    x_test[:,0][x_test[:,0] == -999] = -5
    x_test[:,20][x_test[:,20] == -999] = -5

    x = np.concatenate([x, x**2], axis=1)
    x_test = np.concatenate([x_test, x_test**2], axis=1)

    x, m_train, std_train = scaler(x, train=True)

    x_test, _, _ = scaler(x_test, m=m_train, std=std_train, train=False)

    x = np.hstack((x, np.ones((len(x), 1))))
    x_test = np.hstack((x_test, np.ones((len(x_test), 1))))

    return x, y, x_test, y_test


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/',
                        help='path to the data directory. It should contain `train.csv` and `test.csv` files.')

    args = parser.parse_args()
    data_path = args.data_path

    # submission run
    y_test, x_test, ids_test = load_csv_data(os.path.join(data_path, 'test.csv'))
    y, x, ids = load_csv_data(os.path.join(data_path, 'train.csv'))

    x_train, y_train, x_test, y_test = process_data(x, y, x_test, y_test)

    w_init = np.random.normal(0, 1, size=(x_train.shape[1],))
    w, loss = reg_logistic_regression(y_train, x_train, 0.01, w_init, 1000, 0.05)

    pred_val = sigmoid(x_test @ w)
    predictions = np.where(pred_val > 0.5, 1, -1)

    create_csv_submission(ids_test, predictions, "final.csv")