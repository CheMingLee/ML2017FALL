wget -O w2v_model.syn1neg.npy 'https://github.com/CheMingLee/ML2017FALL/releases/download/model/w2v_model.syn1neg.npy'
wget -O w2v_model.wv.syn0.npy 'https://github.com/CheMingLee/ML2017FALL/releases/download/model/w2v_model.wv.syn0.npy'
python hw4_test.py $1 $2
