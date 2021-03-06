from numpy import *
from plot import *
from util import *
from gradientDescent import *
from predict import *
from scipy.optimize import *
import numpy as np

data = loadtxt('data1.txt', delimiter=',')
X = data[:,0:2]
y = data[:,2]
plot(X,y)

theta = np.zeros(X.shape[1] + 1)
dummy = ones(X.shape[0])
processX = column_stack([dummy, X])
alpha = 1
thetaResult = fmin_bfgs(computeCost, theta, args=(processX, y, alpha), fprime=costGradient)
print thetaResult
print predict(thetaResult, processX, y)