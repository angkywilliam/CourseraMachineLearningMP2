from numpy import *
from plot import *
from util import *
from gradientDescent import *
from predict import *
from scipy.optimize import *
from plotBoundary import *
import numpy as np

data = loadtxt('data2.txt', delimiter=',')
X = data[:,0:2]
y = data[:,2]

processX = mapFeature(X[:,0], X[:,1])
theta = np.zeros(processX.shape[1])
alpha = 1
thetaResult = fmin_bfgs(computeCost, theta, args=(processX, y, alpha), fprime=costGradient)
print thetaResult
print predict(thetaResult, processX, y)
plotBoundary(thetaResult, X, y)
