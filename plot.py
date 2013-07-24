from numpy import *
from pylab import *

def plot(X,y):
	pos = where(y == 1)
	neg = where(y == 0)
	scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
	scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
	xlabel('Exam1 Score')
	ylabel('Exam2 Score')
	legend(['Not Admitted', 'Admitted'])
	show()
