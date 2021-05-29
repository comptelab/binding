#!/usr/bin/python2.7
from bump3_model2 import *
from multiprocessing import Value,Lock
from time import sleep
from os import mkdir,chdir
import sys

n_proc = int(sys.argv[1])
print n_proc

models=[Model(gEEA=0.09,gEIA=0.256,gEIN=0.11,gEEN=0.24,gIE=3,gextI=2.74,gextE=3.5,gII=2,Jp=10,JpIE=2.3,JpEI=2.3,sigmaIE=18,sigmaEI=18,giEEA=0.45,giEIA=0.2) for p in xrange(n_proc)]


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
#models=[Model(gEEA=0.11,gEIA=0.256,gEIN=0.12,gEEN=0.167,gIE=3,gextI=3.74,gextE=5.3,stims=[850,500,100],extra=True) for p in xrange(24)]

