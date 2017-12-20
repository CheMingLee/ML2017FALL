import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Flatten, Dot, Add, Concatenate, Dense, Dropout
from keras.models import Model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import sys

def nn_model(n_users, n_items, latent_dim):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim)(user_input)
    item_vec = Embedding(n_items, latent_dim)(item_input)
    merge_vec = Concatenate()([user_vec, item_vec])
    dropout = Dropout(0.1)(merge_vec)
    dense_layer = Dense(128, activation='relu')(dropout)
    dropout = Dropout(0.1)(dense_layer)
    output = Dense(1, activation='linear')(dropout)
    output = Flatten()(output)
    model = Model([user_input, item_input], output)
    return model

N_users = 6040+1
N_items = 3952+1
latent_Dim = 128
weights_path = 'DNN128.hdf5'
test_path = sys.argv[1]
res_path = sys.argv[2]
movies_path = sys.argv[3]
users_path = sys.argv[4]

print('Load Data...')
test = []
with open(test_path, 'r') as f:
    for line in f:
        line = line.strip('\n')
        line = line.split(',')
        test.append(line)

test = np.array(test)
test = np.delete(test,0,axis = 0)
test = np.array(test, dtype='int64')
Xtst_U = test[:,1]
Xtst_M = test[:,2]

print('Predict...')
model = nn_model(N_users, N_items, latent_Dim)
model.load_weights(weights_path)
pre = model.predict([Xtst_U, Xtst_M])

print('Output...')
with open(res_path, 'w') as f:
    f.write('TestDataID,Rating\n')
    for i,v in enumerate(pre):
        if v < 1:
            f.write('%d,%.3f\n' %(i+1, 1.0))
        elif v > 5:
            f.write('%d,%.3f\n' %(i+1, 5.0))
        else:
            f.write('%d,%.3f\n' %(i+1, v))

print('Done!')
