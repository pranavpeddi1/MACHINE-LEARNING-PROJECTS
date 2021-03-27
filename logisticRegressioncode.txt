import numpy as np
import pandas as pd
import math
data = pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
data.head()
dataset = data.to_numpy()
#print(dataset.shape)

X = dataset[:,0:5]
m = X.shape[0]
y = dataset[:,5]
y = y.reshape(m,1)
Xmean = np.mean(X,0)
Xstd = np.std(X,0)
def fsn(X,Xmean,Xstd):
    Xs = (X-Xmean)/Xstd
    return Xs

Xs = fsn(X,Xmean,Xstd)
#print(Xs)

#print(m)
Xs = np.insert(Xs,0,np.ones((1,m)),axis=1)
#print(Xs)

def hypothesis(Xs,theta):
    tmp = Xs@theta
    yest = 1/(1+np.exp(-tmp))
    return yest

theta = np.array([[0],[0],[0],[0],[0],[0]])
#yest = hypothesis(Xs,theta)
#print(yest)

def cost(Xs,y,theta):
    yest = hypothesis(Xs,theta)
    m = yest.shape[0]
    lh1 = np.log(yest)
    lh2 = np.log(1-yest)
    cs = -(lh1.T@y + lh2.T@(1-y))/m
    return cs

cs = cost(Xs,y,theta)
#print(cs)

def gradient(Xs,y,theta):
    gr = np.zeros((6,1))
    yest = hypothesis(Xs,theta)
    m = yest.shape[0]
    err = yest-y
    gr = Xs.T@err/m
    return gr

N = 300
alpha = 0.1
prevcost = cost(Xs,y,theta)
for i in range(0,N):
    #print(prevcost)
    theta = theta - alpha*gradient(Xs,y,theta)
    cs = cost(Xs,y,theta)
    if abs(prevcost-cs) < 1e-4:
        #print(i)
        break
    prevcost = cs
#print(theta)
#theta = np.linalg.inv(Xs.T@Xs)@Xs.T@y
#print(cost(Xs,y,theta))
#print(theta)
ypred = hypothesis(Xs,theta)
ypred = (np.sign(ypred-0.5)+1)/2
ycomp = np.concatenate((y,ypred),axis=1)
np.set_printoptions(threshold=np.inf)
#print(ycomp)
print(100-np.sum(np.abs(ypred-y))*100/m)
truepositives = 0
falsepositives = 0
falsenegatives = 0
for i in range(0,m):
    if y[i] == 1 and ypred[i] == 1:
        truepositives += 1
    if y[i] == 0 and ypred[i] == 1:
        falsepositives += 1
    if y[i] == 1 and ypred[i] == 0:
        falsenegatives += 1
P = truepositives/(truepositives+falsepositives)
R = truepositives/(truepositives+falsenegatives)
F1 = 2*P*R/(P+R)
#print(m)
#print(truepositives)
#print(falsepositives)
#print(falsenegatives)
print(P)
print(R)
print(F1)

