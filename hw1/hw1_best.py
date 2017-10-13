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

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)
pr.fit(X_quad, y)

tData = pd.read_csv(Test_D, encoding='big5', names=range(11))
T = tData[tData[1] == 'PM2.5']
index_result = pd.DataFrame(np.array(T.iloc[:,0]))
index_result.columns = ['id']
Xt = pd.DataFrame(np.array(T.iloc[:,11-variables:],float))
Xt = Xt.values
result = pr.predict(quadratic.fit_transform(Xt))
result = pd.DataFrame(result)
result.columns = ['value']
outputFile = index_result.join(result)
outputFile.to_csv(Result_name, index=False)
