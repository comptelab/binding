from sklearn import mixture
#!/usr/bin/python2.7
from matplotlib.pylab import *
import numpy as np
import scipy.io
import pylab as pl
from numpy import *
import matplotlib.pyplot as plt
import sys
from pickle import dump
from math import atan2
from joblib import Parallel, delayed
import multiprocessing
from numpy.fft import fft, ifft
num_cores = multiprocessing.cpu_count()
from scipy.signal import butter, lfilter, hilbert



theta = []
r = []
g = mixture.GMM(n_components=2)
#g = mixture.GaussianMixture(n_components=2)
C1=[]
C2=[]
stims=[]

NE = 2048
runtime = 3.5
bn=0.005
stim1_off = 0.75
dt=0.001

time = linspace(0,runtime,runtime/dt)


cue_off = find(time<0.750)[-1]+1
cue_off = 100



def one_sim(f):
	print f
	(spikesE1,spikesI1,sptimesE1,sptimesI1,counts1,spikesE2,spikesI2,sptimesE2,sptimesI2,counts2,params)=load(f)

	spikesE1=(spikesE1,sptimesE1)
	spikesI1=(spikesI1,sptimesI1)
	counts1=counts1

	spikesE2=(spikesE2,sptimesE2)
	spikesI2=(spikesI2,sptimesI2)
	counts2=counts2

	spiketimes1 = array(spikesE1[1].values())
	spiketimes2 = array(spikesE2[1].values())

	bumpA1 = spiketimes1[2*NE/3:NE]
	bumpB1 = spiketimes1[0:NE/3]

	bumpA1 = sorted(reduce(lambda x,y: x+list(y),bumpA1,[]))
	bumpB1 = sorted(reduce(lambda x,y: x+list(y),bumpB1,[]))
	histA1 = histogram(bumpA1, arange(0,runtime,bn)) 
	histB1 = histogram(bumpB1, arange(0,runtime,bn)); 

	bumpA2 = spiketimes2[2*NE/3:NE]
	bumpB2 = spiketimes2[0:NE/3]

	
	bumpA2 = sorted(reduce(lambda x,y: x+list(y),bumpA2,[]))
	bumpB2 = sorted(reduce(lambda x,y: x+list(y),bumpB2,[]))
	histA2 = histogram(bumpA2, arange(0,runtime,bn)); 
	histB2 = histogram(bumpB2, arange(0,runtime,bn)); 


	w=20
	w2=1

	# hA1=histA1[0][histA1[1][1:] > stim1_off]
	# hB1=histB1[0][histB1[1][1:] > stim1_off]


	# hA2=histA2[0][histA2[1][1:] > stim1_off]
	# hB2=histB2[0][histB2[1][1:] > stim1_off]
	
	hA1=histA1[0][histA1[1][1:] > 0]
	hB1=histB1[0][histB1[1][1:] > 0]


	hA2=histA2[0][histA2[1][1:] > 0]
	hB2=histB2[0][histB2[1][1:] > 0]


	cs1=[]
	cs2=[]
	for i in range(0,len(hA1)-15,w2):
		framein1=hA1[i:i+w]
		framein2=hB2[i:i+w]
		cs1+=[corrcoef(framein1,framein2)[0][1]]
		frame11=hB1[i:i+w]
		frame22=hB2[i:i+w]
		cs2+=[corrcoef(frame11,frame22)[0][1]]

	data = [[[n]]*counts2[n] for n in range(NE)]
	data_f =[]
	for d in data:
		data_f += d

	g.fit(data_f) 

	idx = g.weights_ == max(g.weights_)
	theta = g.means_[idx][0][0]



	return cs1, cs2, theta


all_sims = Parallel(n_jobs=num_cores)(delayed(one_sim)(f) for  f in sys.argv[1:])


CS1 = []
CS2 = []
theta = []
for cs1, cs2, t in all_sims:
	CS1.append(cs1)
	CS2.append(cs2)
	theta.append(t)


data = {"theta": theta, "CS1": CS1, "CS2": CS2}
scipy.io.savemat("CS_data.mat",data)

