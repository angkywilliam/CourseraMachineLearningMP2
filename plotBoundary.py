from numpy import *
from pylab import *
from util import *
def plotBoundary(theta, X, y):
	pos = where(y == 1)
	neg = where(y == 0)
	scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
	scatter(X[neg, 0], X[neg, 1], marker='x', c='r')

	u = linspace(-1, 1.5, 50)
	v = linspace(-1, 1.5, 50)
	z = zeros(shape=(len(u), len(v)))

	for i in range(len(u)):
		for j in range(len(v)):
			z[i, j] = (mapFeature(array(u[i]), array(v[j])).dot(array(theta)))
 	z = z.T
 	contour(u,v,z)
	xlabel('Observe1')
	ylabel('Observe2')
	legend(['y = 1', 'y = 0'])
	show()
