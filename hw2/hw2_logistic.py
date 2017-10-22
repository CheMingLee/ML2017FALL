import pandas as pd
import numpy as np
from math import floor
import sys

raw_data = sys.argv[1]
test_data = sys.argv[2]
train_data_path = sys.argv[3]
train_label_path = sys.argv[4]
test_data_path = sys.argv[5]
output_path = sys.argv[6]

def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path)
    X_train = X_train.values
    Y_train = pd.read_csv(train_label_path)
    Y_train = Y_train.values
    X_test = pd.read_csv(test_data_path)
    X_test = X_test.values

    return X_train, Y_train, X_test

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def Rescaling(X_all, X_test):
    # Feature Rescaling with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    X_max = X_train_test.max(axis=0)
    X_min = X_train_test.min(axis=0)
    X_train_test_rescaled = (X_train_test-X_min)/(X_max-X_min)

    # Split to train, test again
    X_all = X_train_test_rescaled[0:X_all.shape[0]]
    X_test = X_train_test_rescaled[X_all.shape[0]:]
    return X_all, X_test

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]

    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-11, 1-(1e-11))

def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)
    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))

def train(X_all, Y_all):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)

    # Initiallize parameter, hyperparameter
    fnumber = len(X_train[0])
    w = np.zeros(fnumber)
    b = 0
    l_rate = 1
    b_lr = 0
    w_lr = np.zeros(fnumber)
    batch_size = len(X_train)
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = int(1e4+1)
    save_param_iter = int(1e3)

    # Start training
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        # Do validation
        if (epoch) % save_param_iter == 0:
            print('=====Param at epoch %d=====' % epoch)
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)
        
        # Random shuffle
        X_train, Y_train = _shuffle(X_train, Y_train)

        # Train with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, w) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = -np.dot(X.T, (np.squeeze(Y) - y))
            b_grad = -np.sum(np.squeeze(Y)-y)

            # adagrad
            b_lr += b_grad**2
            w_lr += w_grad**2
            w = w - l_rate/np.sqrt(w_lr) * w_grad
            b = b - l_rate/np.sqrt(b_lr) * b_grad
    
    return w, b

def infer(X_test, w, b, output_path):
    test_data_size = len(X_test)

    # predict
    z = np.dot(X_test, w) + b
    y = sigmoid(z)
    y_ = np.around(y)

    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

X_all, Y_all, X_test = load_data(train_data_path, train_label_path, test_data_path)

# Standardization
# X_all, X_test = normalize(X_all, X_test)
# Rescaling
X_all, X_test = Rescaling(X_all, X_test)

# train and infer
w, b = train(X_all, Y_all)
infer(X_test, w, b, output_path)
