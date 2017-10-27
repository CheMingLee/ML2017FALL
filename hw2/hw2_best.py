import pandas as pd
import numpy as np
from math import floor
import sys
from sklearn.svm import SVC

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
    Y_train = np.squeeze(Y_train)
    Y_valid = np.squeeze(Y_valid)
    return X_train, Y_train, X_valid, Y_valid

def train(model, X_all, Y_all):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    model.fit(X_train, Y_train)
    print('validation acc = %f' %model.score(X_valid, Y_valid))

def infer(model, X_test, output_path):
    result = model.predict(X_test)    
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(result):
            f.write('%d,%d\n' %(i+1, v))

X_all, Y_all, X_test = load_data(train_data_path, train_label_path, test_data_path)
# Standardization
# X_all, X_test = normalize(X_all, X_test)
# Rescaling
X_all, X_test = Rescaling(X_all, X_test)
# train and infer
svm = SVC(kernel='linear', C=1.0, random_state=0)
# svm
print('=====SVM=====')
train(svm, X_all, Y_all)
infer(svm, X_test, output_path)
