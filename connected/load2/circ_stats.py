from scipy.stats import circmean,circvar
from cmath import phase
from  numpy import array
from scipy.stats import circmean,circvar,circstd
from numpy import *
from cmath import phase
from matplotlib.pylab import *


def len2(x):
	if type(x) is not type([]):
		if type(x) is not type(array([])):
			return -1
	return len(x)

def phase2(x):
	if not isnan(x):
		return phase(x)
	return nan

def circdist(angles1,angles2):
	if len2(angles2) < 0:
		if len2(angles1) > 0:
			angles2 = [angles2]*len(angles1)
		else:
			angles2 = [angles2]
			angles1 = [angles1]		
	if len2(angles1) < 0:
		angles1 = [angles1]*len(angles2)
	return amap(lambda a1,a2: phase2(exp(1j*a1)/exp(1j*a2)), angles1,angles2)

def circdist_2pi(angles1,angles2):
	dist = circdist(angles1,angles2)
	dist[dist<0]+=2*pi
	return dist