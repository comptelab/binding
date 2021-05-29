from bump1_model import *
from multiprocessing import Value,Lock
from time import sleep

n_runs = 2

models=[Model(gEEA=0.126,gEIA=0.256,gEIN=0.11,gEEN=0.2,gIE=3,gextI=2.8,gextE=3.58,gII=2,runtime=5,Jp=11,JpIE=2.6,JpEI=2.6,sigmaIE=30,sigmaEI=30,stim=3) for p in xrange(n_runs)]


map(lambda m: m.start(),models)
map(lambda m: m.join(),models)
map(lambda m: m.plot(),models)


# Model 0
# Model() <- presistent activity, but no oscilations (no AMPA)

# Model 1
# Presistent activity with some oscilations
# too high rates
#models=[Model(gEEA=0.06,gEIA=0.233,gEIN=0.15,gEEN=0.476,sigma=9) for p in xrange(24)]

# Model 2
# Less times presistent, but when it is presistent oscilations look good
# still with too high rates 100-200 Hz
#gEEN=linspace(0.476,0.4,24)
#models=[Model(gEEA=0.1,gEIA=0.233,gEIN=0.1,gEEN=gEEN[p],sigma=9) for p in xrange(24)]

# Model 3


# Model 4
# Three bumps, presistent and low rates (arround 60-80hz) and nice oscilations
# some times the bump explodes and comes back later
#models=[Model(gEEA=0.11,gEIA=0.256,gEIN=0.12,gEEN=0.167,gIE=3,gextI=3.74,gextE=5.3,stims=[850,500,150],extra=True) for p in xrange(24)]

