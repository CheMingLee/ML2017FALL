import pandas as pd
import numpy as np

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

w = np.zeros(len(X[0]))
l_rate = 10
repeat = 10000
lamda = 1e-1
x_t = X.transpose()
s_gra = np.zeros(len(X[0]))

for i in range(repeat):
    hypo = np.dot(X,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(X)
    cost_a  = np.sqrt(cost)
    gra = np.dot(x_t,loss)+lamda*w
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    # print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

# save model
np.save('PM25SquareLamda0.npy',w)
