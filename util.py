from numpy import *

def sigmoid(X):
	return 1/(1 + e ** (-X))

def computeCost(theta, X, y):
	hypothesis = sigmoid( dot(X, theta) )
	tempResult = (-y * log(hypothesis)) - (1 - y) * log(1 - hypothesis)
	tempResult = sum(tempResult)
	cost = tempResult / X.shape[0]
	return cost

def costGradient(theta, X, y):
	hypothesis = sigmoid( dot(X, theta) )
	grad = dot( (hypothesis - y).T, X) / X.shape[0]
	return grad.T