from util import *
from collections import namedtuple

def gradientDescent(initTheta, X, y, maxIter, alpha):
	costRecord = []
	theta = initTheta
	Tuple = namedtuple('Point', 'theta costRecord')

	for i in range (0, maxIter):
		curCost = computeCost(theta, X, y)
		costRecord.append(curCost)
		grad = costGradient(theta, X, y)
		theta = theta - (alpha * grad)
		#print theta
		#raw_input("Press Enter to continue...")
	
	result = Tuple(theta, costRecord)
	return result