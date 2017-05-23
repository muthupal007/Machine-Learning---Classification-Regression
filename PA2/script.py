import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD

    out_classes = np.unique(y)
    in_classes = np.zeros((len(out_classes)), dtype=np.ndarray)
    means = np.zeros((len(X.T),len(out_classes))) #d X k

    i = 0
    for out_class in out_classes:
        in_classes[i]  = X[np.argwhere(y==out_class).T[0]]
        means.T[i] = np.mean(in_classes[i],axis=0)
        i += 1
    covmat = np.cov(X.T)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD

    out_classes = np.unique(y)
    in_classes = np.zeros((len(out_classes)), dtype=np.ndarray)
    means = np.zeros((len(X.T),len(out_classes))) #d X k
    covmats = np.zeros((len(out_classes),len(X.T),len(X.T)))

    i = 0
    for out_class in out_classes:
        in_classes[i]  = X[np.argwhere(y==out_class).T[0]]
        means.T[i] = np.mean(in_classes[i], axis=0)
        covmats[i] = np.cov(in_classes[i].T)
        i += 1
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    ypred=np.zeros((len(Xtest)))
    pred = np.zeros((len(means.T)), dtype=np.ndarray)

    i = 0
    count = 0

    for x in Xtest:
        j = 0
        for mean in means.T:
            x_sub_mu = np.subtract(x, mean)
            cov_inv = np.linalg.inv(covmat)
            pred[j] = np.exp(-0.5 * np.dot(np.dot(x_sub_mu.T,cov_inv).T,x_sub_mu))
            j += 1
        ypred[i] = np.argmax(pred) + 1
        if ypred[i] == ytest[i][0]:
            count += 1
        i += 1
    acc = (count/len(ytest)) * 100
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    ypred=np.zeros((len(Xtest)))
    pred = np.zeros((len(means.T)), dtype=np.ndarray)

    i = 0
    count = 0

    for x in Xtest:
        #for mean in means.T:
        for j in range (0,len(means.T)):
            mean = means.T[j]
            covmat = covmats[j]
            x_sub_mu = np.subtract(x, mean)
            cov_inv = np.linalg.inv(covmat)
            cov_det = np.linalg.det(covmat)
            pred[j] = np.exp(-0.5 * np.dot(np.dot(x_sub_mu,cov_inv).T,x_sub_mu)) / np.power(cov_det,0.5)
        ypred[i] = np.argmax(pred) + 1
        if ypred[i] == ytest[i][0]:
            count = count + 1
        i = i + 1
    acc = (count/len(ytest)) * 100
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD     

    w = np.dot(X.T, X)
    w = np.linalg.inv(w)
    w = np.dot(w, X.T)
    w = np.dot(w, y)                                            
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD    '
    w = np.dot(X.T, X)
    temp = lambd * np.identity(len(X.T))    
    w = np.add(temp, w)
    w = np.linalg.inv(w)
    w = np.dot(w, X.T)
    w = np.dot(w, y)                                          
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    N = len(Xtest)
    temp = np.dot(Xtest,w)
    temp = np.subtract(ytest,temp)
    temp = np.square(temp)
    mse = np.sum(temp)/N
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD    
    w = np.reshape(w,[len(X.T),1])
    t2 = np.dot(w.T, w)
    t2 = (lambd * t2)/2
    t1 = np.dot(X, w)
    t1 = np.subtract(y, t1)
    t1 = np.square(t1)
    t1 = np.sum(t1)/2
    error = np.add(t1,t2)

    
    t4 = lambd * w
    t3 = np.dot(X, w)
    t3 = np.subtract(t3,y)
    t3 = np.dot(X.T, t3)
    error_grad = np.add(t3,t4)      
    error_grad = error_grad.flatten()
                                     
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
	
    # IMPLEMENT THIS METHOD
    Xd = np.zeros([len(x),p+1])
    for i in range(0,p+1):
        Xd[:,i] = np.power(x,i)
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()

# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mlet = testOLERegression(w, X, y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mlet_i = testOLERegression(w_i,X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept for training data '+str(mlet))
print('MSE with intercept for training data '+str(mlet_i))

print('MSE without intercept for testing data '+str(mle))
print('MSE with intercept for testing data '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

opt_lambda = mses3.argmin() * 0.01;
print("Optimal value of lambda",opt_lambda)
mses3t_min = min(mses3_train)
mses3_min = min(mses3)
print('MSE for training data using Ridge Regression', mses3t_min)
print('MSE for testing data using Ridge Regression', mses3_min)
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 50}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

mses4t_min = min(mses4_train)
mses4_min = min(mses4)
print('MSE for training data using Gradient Descent for Ridge Regression', mses4t_min)
print('MSE for testing data using Gradient Descent for Ridge Regression', mses4_min)
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = opt_lambda # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

popt1 = mses5[:,0].argmin()
popt2 = mses5[:,1].argmin()
popt1t = mses5_train[:,0].argmin()
popt2t = mses5_train[:,1].argmin()
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
print("Optimal p without Regularization for training data: ",popt1t)
print("Optimal p with Regularization for training data: ",popt2t)
print("Optimal p without Regularization for testing data: ",popt1)
print("Optimal p with Regularization for testing data: ",popt2)
print('MSE for training data using Non Linear Regression without Regularization', min(mses5_train[:,0]))
print('MSE for training data using Non Linear Regression with Regularization', min(mses5_train[:,1]))
print('MSE for test data using Non Linear Regression without Regularization', min(mses5[:,0]))
print('MSE for test data using Non Linear Regression with Regularization', min(mses5[:,1]))
