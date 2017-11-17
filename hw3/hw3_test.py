import numpy as np
import pandas as pd
from keras.models import load_model
import h5py
import sys

test_data = pd.read_csv(sys.argv[1])
test_data = np.array(test_data)
test_feature = []
for i in range(len(test_data)):
    test_feature.append(np.reshape(np.array(test_data[i,1].split(' ')), (48,48)))
    
test_feature = np.array(test_feature, float)
test_feature = test_feature.reshape(len(test_data),48,48,1)
test_feature = test_feature/255

# fit
model = load_model('DataGen_model2.h5')
prediction = model.predict(test_feature)
predict_result = np.argmax(prediction, axis=1)

# output
file_write_result = open(sys.argv[2], 'w')
file_write_result.write('id,label\n')
for i in range(len(test_feature)):
	write_string = str(i) + ',' + str(predict_result[i]) + '\n'
	file_write_result.write(write_string)
