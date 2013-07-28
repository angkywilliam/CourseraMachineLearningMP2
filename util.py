from numpy import *

def sigmoid(X):
	return 1/(1 + e ** (-X))

def computeCost(theta, X, y, alpha):
	hypothesis = sigmoid( dot(X, theta) )
	tempResult = (-y * log(hypothesis)) - (1 - y) * log(1 - hypothesis)
	tempResult = sum(tempResult)
	cost = tempResult / X.shape[0]

	newTheta = theta
	newTheta[0] = 0
	sqTheta = newTheta * newTheta
	cost = cost + (sum(sqTheta) * alpha) / (2 * X.shape[0])
	return cost

def costGradient(theta, X, y, alpha):
	hypothesis = sigmoid( dot(X, theta) )
	gradTemp = dot( (hypothesis - y).T, X)
	newTheta = theta
	newTheta[0] = 0
	grad = gradTemp + (newTheta * alpha).T
	grad = grad / X.shape[0]
	return grad.T

def mapFeature(X1, X2):
	out = ones(X1.size)
	degree = 6
	for i in range( 1,degree + 1):
		for j in range(0, i + 1):
			temp = ( X1 ** (i-j) ) * (X2 ** j)
			out = column_stack((out,temp))
	return out	 