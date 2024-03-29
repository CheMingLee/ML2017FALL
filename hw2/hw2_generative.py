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
    return np.clip(res, 1e-10, 1-(1e-10))

def valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))

def train(X_all, Y_all):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    # Gaussian distribution parameters
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0
    mu1 = np.zeros((106,))
    mu2 = np.zeros((106,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((106,106))
    sigma2 = np.zeros((106,106))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
    N1 = cnt1
    N2 = cnt2
    print('=====Validating=====')
    valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)
    return mu1, mu2, shared_sigma, N1, N2

def infer(X_test, mu1, mu2, shared_sigma, N1, N2, output_path):
    # Predict
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot(mu1, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2, sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    result = np.around(y)
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(result):
            f.write('%d,%d\n' %(i+1, v))

X_all, Y_all, X_test = load_data(train_data_path, train_label_path, test_data_path)
# Standardization
X_all, X_test = normalize(X_all, X_test)
# Rescaling
# X_all, X_test = Rescaling(X_all, X_test)
# train and infer
mu1, mu2, shared_sigma, N1, N2 = train(X_all, Y_all)
infer(X_test, mu1, mu2, shared_sigma, N1, N2, output_path)
