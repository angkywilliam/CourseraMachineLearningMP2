from numpy import *
from plot import *
from util import *
from gradientDescent import *
from collections import namedtuple


data = loadtxt('data1.txt', delimiter=',')
X = data[:,0:2]
y = data[:,2]
#plot(X,y)

theta = zeros(X.shape[1] + 1)
dummy = ones(X.shape[0])
processX = column_stack([dummy, X])
result = gradientDescent(theta, processX, y, 400, 0.01)