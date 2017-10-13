import pandas as pd
import numpy as np
import sys

Train_D = './data/train.csv'
Test_D = sys.argv[1]
Result_name = sys.argv[2]

def reshapeData(allData, featureName, months):
    V = allData[allData['測項'] == featureName]
    V = V.drop(['日期', '測站', '測項'], axis=1)
    v = pd.DataFrame(np.array(V,float).reshape((months,-1)))
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

xfit_ = []
xvalid_ = []
yfit_ = []
yvalid_ = []
for i in range(len(X)):
    if i%4 == 2:
        xvalid_.append(X[i])
        yvalid_.append(y[i])
    else:
        xfit_.append(X[i])
        yfit_.append(y[i])

X = np.array(xfit_)
Xvalidation = np.array(xvalid_)
y = np.array(yfit_)
y_validation = np.array(yvalid_)

eta = 1
B_eta = 0
W_eta = np.zeros(variables)
W2_eta = np.zeros(variables)
Bias = 2
Weight = np.zeros(variables)
Weight2 = np.ones(variables)
RMSE_ = []
itera = 20000
for _ in range(itera):
    L = y-(Bias+np.dot(X,Weight)+np.dot(X**2,Weight2))
    B_grad = -L.sum()*2
    W_grad = -np.dot(X.T,L)
    W2_grad = -np.dot((X**2).T,L)
    B_eta += B_grad**2
    W_eta += W_grad**2
    W2_eta += W2_grad**2
    Bias += (-eta/np.sqrt(B_eta))*B_grad
    Weight += (-eta/np.sqrt(W_eta))*W_grad
    Weight2 += (-eta/np.sqrt(W2_eta))*W2_grad
    RMSE = np.sqrt((L**2).sum()/len(L))
    RMSE_.append(RMSE)

Loss = y_validation-(Bias+np.dot(Xvalidation,Weight)+np.dot(Xvalidation**2,Weight2))
RMSEvalidation = np.sqrt((Loss**2).sum()/len(Loss))

print('b: {0}'.format(Bias))
print('w1: {0}'.format(Weight))
print('w2: {0}'.format(Weight2))
print('RMSE(self): {0}'.format(RMSE))
print('RMSE(validation): {0}'.format(RMSEvalidation))

tData = pd.read_csv(Test_D, encoding='big5', names=range(11))
T = tData[tData[1] == 'PM2.5']
index_result = pd.DataFrame(np.array(T.iloc[:,0]))
index_result.columns = ['id']
Xt = pd.DataFrame(np.array(T.iloc[:,11-variables:],float))
Xt = Xt.values
result = Bias+np.dot(Xt,Weight)+np.dot(Xt**2,Weight2)
result = pd.DataFrame(result)
result.columns = ['value']
outputFile = index_result.join(result)
outputFile.to_csv(Result_name, index=False)
