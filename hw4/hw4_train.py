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

train_label_path = sys.argv[1]
train_nolabel_path = sys.argv[2]
w2vmodel_path = 'w2v_model'
weights_path = 'best_hw4_model.hdf5'
max_len = 30
emb_dim = 300

f_trn = open(train_label_path, 'r', encoding='utf8')
trn_txt = f_trn.readlines()
Ytrn = [int(i.split(' +++$+++ ')[0]) for i in trn_txt]
Xtrn = [i.split(' +++$+++ ')[1].split(' ') for i in trn_txt]

f_pre_trn = open(train_nolabel_path, 'r', encoding='utf8')
pre_trn_txt = f_pre_trn.readlines()
Xpre_trn = [i.split(' ') for i in pre_trn_txt]

w2vmodel = gensim.models.Word2Vec.load(w2vmodel_path)
Xtrn = w2v_processing(w2vmodel, Xtrn, max_len, emb_dim)

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
model.summary()
adam = Adam(lr=0.001,decay=1e-6)
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(filepath=weights_path,
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='auto')

model.fit(Xtrn, Ytrn, validation_split=0.1, batch_size=64, epochs=20, callbacks=[earlystopping, checkpoint])
