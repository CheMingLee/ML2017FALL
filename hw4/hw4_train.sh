wget -O w2v_model.syn1neg.npy 'https://www.dropbox.com/s/fod2w7hb7bpdw8u/w2v_model.syn1neg.npy?dl=1'
wget -O w2v_model.wv.syn0.npy 'https://www.dropbox.com/s/agtcw63bu71zk4y/w2v_model.wv.syn0.npy?dl=1'
python hw4_train.py $1 $2
