from numpy import *
from util import *

def predict(theta, X, y):
	result = sigmoid(dot(X,theta)) >= 0.5
	return float(sum(result == y)) / float(y.shape[0])