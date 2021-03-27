import numpy as np
import pandas as pd

data = pd.read_csv('../input/real-estate-price-prediction/Real estate.csv')
data.head()
#print(type(data))
dataset = data.to_numpy()
#print(type(dataset))
X = dataset[:,1:7]
#print(X.shape)
#print(X)
y = dataset[:,7]
#print(y.shape)
m = y.shape[0]
y = y.reshape(m,1)
#print(y.shape)
#print(y)

X = np.insert(X,0,np.ones((1,414)),axis=1)
#print(X.shape)
def hypothesis(X,theta):
    y = X@theta
    return y

theta = np.array([[0.1],[0],[-0.0004],[0],[-0.002],[0],[0.001]])
#print(theta)
#yest = hypothesis(X,theta)
#print(yest.shape)

#print(hypothesis(X,theta))
def cost(X,y,theta):
    yest = hypothesis(X,theta)
    err = yest-y
    m = err.shape[0]
    cs = err.T @ err/(2*m)
    return cs

#cs = cost(X,y,theta)
#print(cs.shape)

def gradient(X,y,theta):
    gr = np.zeros((7,1))
    yest = hypothesis(X,theta)
    err = yest-y
    #print(err.shape)
    #m = err.shape[0]
    gr = X.T @ err/m
    return gr

#gradient(X,y,theta)
N = 300
alpha = 2e-7
prevcost = cost(X,y,theta)
for i in range(0,N):
    print(prevcost)
    theta = theta - alpha*gradient(X,y,theta)
    cs = cost(X,y,theta)
    if abs(prevcost-cs) < 1e-4:
        print(i)
        break
    prevcost = cs
print(theta)
theta = np.linalg.inv(X.T@X)@X.T@y
print(cost(X,y,theta))
print(theta)

