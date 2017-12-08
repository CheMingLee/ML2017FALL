import gensim
from gensim.models import word2vec
import logging
import pandas as pd 
import numpy as np
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import LSTM, GRU, Dropout, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
import sys

def w2v_processing(w2vmodel,x,mlen,dim):
    empty_word = np.zeros(dim).astype(float)

    X = np.zeros((len(x),mlen,dim))

    for i,post in enumerate(x):
        for j,word in enumerate(post):
            if j == mlen:
                break
            else:
                if word in w2vmodel:
                    X[i,j,:] = w2vmodel[word]
                else:
                    X[i,j,:] = empty_word
    return X

test_path = sys.argv[1]
res_path = sys.argv[2]
w2vmodel_path = 'w2v_model'
weights_path = 'best_hw4_label.hdf5'
max_len = 30
emb_dim = 300

f_tst = open(test_path, 'r', encoding='utf8')
tst_txt = f_tst.readlines()
Xtst = [i.split(',',1)[1].split(' ') for i in tst_txt]
Xtst = Xtst[1:]

w2vmodel = gensim.models.Word2Vec.load(w2vmodel_path)
Xtst = w2v_processing(w2vmodel, Xtst, max_len, emb_dim)

print('Creating a best model...')
model = Sequential()
model.add(GRU(256,activation ='tanh',
                recurrent_initializer = 'orthogonal',
                bias_initializer='ones',
                recurrent_dropout=0.1,
                return_sequences=True,
                dropout=0.1,input_shape=(max_len,emb_dim)))
model.add(GRU(128,activation ='tanh',
                recurrent_initializer = 'orthogonal',
                bias_initializer='ones', 
                recurrent_dropout=0.1,
                dropout=0.1))
model.add(Dense(16,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.load_weights(weights_path)

print('Predicting...')
result = model.predict(Xtst)
x_id = np.arange(0,len(Xtst))[:,np.newaxis]
y_ans = np.zeros(len(x_id),int)

for i in range(len(y_ans)):
    if result[i]<0.5:
        y_ans[i]=0
    else:
        y_ans[i]=1

x_id = np.append(['id'], x_id)
y_ans = np.append(['label'], y_ans)
df = pd.DataFrame(y_ans, x_id)
df.to_csv(res_path, header=None)

print('Finish!')
