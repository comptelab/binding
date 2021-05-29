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

tau=5;
g_max=1;
#t_syn = 0:1:ceil(20*tau);
t_syn = linspace(0,100,100)
alpha_syn = g_max * t_syn/tau * exp(-t_syn/tau);
alpha_syn = alpha_syn/max(alpha_syn);

# check spikes
def convert_to_lfp(spikes):
	neurons_i =spikes[:,0]
	sp_t_i = amap(int,spikes[:,1]/dt)
	spikes = zeros((2048,(int(3.5/dt)+1)))
	for ni in range(NE):
		spikes[ni,sp_t_i[neurons_i == ni]] = 1
	lfpa = mean(spikes[2*NE/3:NE,:],0)
	lfpb = mean(spikes[0:NE/3],0)

	return convolve(alpha_syn,lfpa),convolve(alpha_syn,lfpb)

def one_sim(f):
	print f
	(spikesE1,spikesI1,sptimesE1,sptimesI1,counts1,spikesE2,spikesI2,sptimesE2,sptimesI2,counts2,params)=load(f)

	data = [[[n]]*counts2[n] for n in range(NE)]
	data_f =[]
	for d in data:
		data_f += d

	g.fit(data_f) 

	idx = g.weights_ == max(g.weights_)
	theta = g.means_[idx][0][0]


	lfp1 = convert_to_lfp(array(spikesE1))

	lfp2 = convert_to_lfp(array(spikesE2))

	return theta,lfp1,lfp2


all_sims = Parallel(n_jobs=num_cores)(delayed(one_sim)(f) for  f in sys.argv[1:])


LFP1 = []
LFP2 = []
theta = []
for t, lfp1,lfp2 in all_sims:
	theta.append(t)
	LFP1.append(lfp1)
	LFP2.append(lfp2)

data = {"theta": theta, "lfp1": LFP1, "lfp2": LFP2}
scipy.io.savemat("lfp_data.mat",data)

k

f=open("oscilation_data.pickle","w")
dump(all_sims,f)
f.close()

subplot(2,1,1)
title("no swaps")
plot(mean(array(LFP1)[array(theta)<600],0).T)
xlim(800,1500)
ylim(0.04,0.06)

subplot(2,1,2)
title("swaps")
plot(mean(array(LFP1)[array(theta)>600],0).T)
xlim(800,1500)
ylim(0.04,0.06)