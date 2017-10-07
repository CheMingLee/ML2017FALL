import pandas as pd
import numpy as np
from numpy.linalg import inv

Train_D = './data/train.csv'

def reshapeData(allData, featureName, months):
    V = allData[allData['測項'] == featureName]
    V = V.drop(['日期', '測站', '測項'], axis=1)
    v = pd.DataFrame(np.array(V, float).reshape((months,-1)))
    return v

def features(v, X, variables, dataPerM):
    for i in range(dataPerM-variables):
        df = v.iloc[:,i:i+variables]
        df.columns = np.array(range(variables))
        X.append(df)
    return X

def targets(v, y, variables, dataPerM):
    for i in range(variables, dataPerM):
        df = v.iloc[:,i]
        df.columns = np.array([0])
        y.append(df)
    return y

allData = pd.read_csv(Train_D, encoding='big5')
variables = 9
months = 12
days = 20
hours = 24
dataPerM = days*hours
X_ = []
y_ = []

featureName_ = ['PM2.5']
for featureName in featureName_:
    v = reshapeData(allData, featureName, months)
    X_ = features(v, X_, variables, dataPerM)
    y_ = targets(v, y_, variables, dataPerM)
X = pd.concat(X_, ignore_index=True)
X = X.values
y = pd.concat(y_, ignore_index=True)
y = y.values

# add square term
X = np.concatenate((X,X**2), axis=1)
# add bias
X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)

# close form
w = np.matmul(np.matmul(inv(np.matmul(X.transpose(),X)),X.transpose()),y)

# save model
np.save('PM25SquareCloseForm.npy',w)
