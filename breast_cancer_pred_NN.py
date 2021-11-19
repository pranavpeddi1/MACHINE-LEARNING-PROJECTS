import numpy as np
import pandas as pd

data = pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
data.head()

dataset = data.to_numpy()
print(dataset.shape)

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
Xs = np.insert(Xs,0,np.ones((1,m)),axis=1)

# num of inputs = num of features + 1 = 6
# num of outputs = 1
# num of hidden layers = 1
# num of neurons in the hidden layer = 3
# activation functions in hidden layer - sigmoid function
# activation function in the output layer - sigmoid function

# 
s1 = 5
s2 = 3
s3 = 1
epsilon = 0.001
Th1 = np.random.rand(3,6)
Th2 = np.random.rand(1,4)
Th1 = Th1*2*epsilon-epsilon
Th2 = Th2*2*epsilon-epsilon
#print(Th1)
#print(Th2)
def g(z):
    return 1/(1+np.exp(-z))
z2 = Th1@Xs.T
a2 = g(z2)
a2 = np.insert(a2,0,np.ones((1,m)),axis=0)
#print(a2)
z3 = Th2@a2
#print(z3.shape)
a3 = g(z3)
print(a3)
