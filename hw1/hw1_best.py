import pandas as pd
import numpy as np
import sys

intput_file = sys.argv[1]
output_file = sys.argv[2]

variables = 9

# read model
w = np.load('PM25SquareCloseForm.npy')

tData = pd.read_csv(intput_file, encoding='big5', names=range(11))
T = tData[tData[1] == 'PM2.5']
index_result = pd.DataFrame(np.array(T.iloc[:,0]))
index_result.columns = ['id']
Xtest = pd.DataFrame(np.array(T.iloc[:,11-variables:],float))
Xtest = Xtest.values

# add square term
Xtest = np.concatenate((Xtest,Xtest**2), axis=1)
# add bias
Xtest = np.concatenate((np.ones((Xtest.shape[0],1)),Xtest), axis=1)

# output
result = np.dot(Xtest,w)
result = pd.DataFrame(result)
result.columns = ['value']
outputFile = index_result.join(result)
outputFile.to_csv(output_file, index=False)
