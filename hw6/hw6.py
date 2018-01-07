import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
import sys

train_path = sys.argv[1]
test_path = sys.argv[2]
res_path = sys.argv[3]

encoding_dim = 16
weights_path = 'model.hdf5'

train = np.load(train_path) / 255.

input_img = Input(shape=(784,))
encoded = Dense(512, activation='relu')(input_img)
encoded = BatchNormalization()(encoded)
encoded = Dense(256, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)

encoder = Model(input_img, encoded)
encoder.load_weights(weights_path)
encoded_train = encoder.predict(train)

test = np.array(pd.read_csv(test_path)).astype(int)
id_test = test[:,0]
img1_id = test[:,1]
img2_id = test[:,2]

k_means = KMeans(n_clusters=2).fit(encoded_train)
test_label = k_means.labels_

ans = np.zeros(len(id_test), int)

with open(res_path, 'w') as f:
	f.write('ID,Ans\n')
	for i in id_test:
		label_img1 = test_label[img1_id[i]]
		label_img2 = test_label[img2_id[i]]
		if label_img1 == label_img2:
			ans[i] = 1

		f.write('{0},{1}\n'.format(i, ans[i]))
