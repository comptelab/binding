from sklearn import mixture
#!/usr/bin/python2.7

import numpy as np
import scipy.io
import pylab as pl
from numpy import *
import matplotlib.pyplot as plt
import sys
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
from math import atan2


theta = []
r = []
g = mixture.GMM(n_components=2)
#g = mixture.GaussianMixture(n_components=2)
C1=[]
C2=[]
stims=[]

for f in sys.argv[1:]:
	f = open(f,'r')
	print f


def one_file(f):
	(spikesE1,spikesI1,sptimesE1,sptimesI1,counts1,spikesE2,spikesI2,sptimesE2,sptimesI2,counts2,params)=load(f)
	
	spikesE1=(spikesE1,sptimesE1)
	spikesI1=(spikesI1,sptimesI1)
	counts1=counts1

	spikesE2=(spikesE2,sptimesE2)
	spikesI2=(spikesI2,sptimesI2)
	counts2=counts2

	[stim1,stim2,stim3] = params["stims"]
	#stims+=[stim2]

	# C1+=[counts1]
	# C2+=[counts2]
	NE=len(counts1)

	data = [[[n]]*counts2[n] for n in range(NE)]
	data_f =[]
	for d in data:
		data_f += d

	g.fit(data_f) 

	idx = g.weights_ == max(g.weights_)
	theta=[g.means_[idx][0][0]]
	return theta,stim2,counts1,counts2

theta = Parallel(n_jobs=num_cores)(delayed(one_file)(f) for  f in sys.argv[1:])

swaps = array(theta) > 600