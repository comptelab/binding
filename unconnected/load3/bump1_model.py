import numpy as np
from scipy import fftpack
import pylab as pl

from brian import *
from numpy.fft import rfft,irfft
from numpy.random import seed
from copy import deepcopy
from multiprocessing import Process,Queue,Pipe
from ctypes import *
from pickle import *
from threading import *
from bisect import insort

# CONDACTANGE (g)
# are associated with the leak, i.e., the pours of the membrane otherwise impervious.
# The only pours taken into account in these models are the recepetors and a general
# previability given by the gLeak associated to each type of neurons.

# CAPACITANCE (C)
# Something like the amount of current the cell is able to contain, if it didn't have 
# any g associated, i.e., pours. Associated with each type of neuron.

CmE=0.5*nF 		# capacitance e-cells
CmI=0.2*nF 		# capacitance i-cells
gLeakE=25*nS 		# leak conductance e-cells
gLeakI=20*nS 		# leak conductance i-cells

taua = 2*ms 		# AMPA synapse decay
taun = 100*ms 		# NMDA synapse decay
taux = 2*ms 		# NMDA synapse rise
taug = 10*ms 		# GABA synapse decay

Vt  =-50*mvolt          # spike threshold
Vr  =-60*mvolt          # reset value
El  =-70*mvolt          # resting potential / holding potential? 
refE= 2*ms              # refractory periods
refI= 1*ms              # refractory periods

Ven = 16.129*mV 	#voltage reference in the NMDA magnesium block equation

#stim1=3*1024/4
#stim2=1024/2
#stim3=1024/4

# three
#stim1=850
#stim2=500
#stim3=150


def cross(bump1,bump2,bn):
        spec=bump1*conj(bump2)
        l=irfft(spec)[-0.2/bn-1:]
        r=irfft(spec)[:0.2/bn+1]
        crossc=list(l)+list(r)
        crossc-=mean(crossc)
        return crossc


class Model(Process):
    # Sets default parameters (no AMPA, no oscilations) if none given
    def __init__(self,on=3000,off=3500,runtime=6.5,
                gEEA=0,gEEN=0.484,gEIA=0,gEIN=0.379,
                gIE=3.643,gII=2.896,gextE=6.6,gextI=5.8,
                sigma= 9,sigmaIE=32.4,sigmaEI=32.4,
                Jp=5.7,JpIE=1.4,JpEI=1.4,
        #       NE=1024,NI=256,stim=1,stims=[0.83*1024,0.48*1024,0.146*1024],extra=False):
                NE=1024,NI=256,stim=1,stims=[3*1024/4,1024/2,1024/4],extra=False):
        #       NE=1024,NI=256,stim=1,stims=[850,0,150],extra=False):

        Process.__init__(self)
	self.extra = extra
	self.stim = stim
	self.load=False
	self.spikesI=None
	self.spikesE=None
	self.counts=None
	self.phase=None
        self.stim_on=on*ms
        self.stim_off=off*ms
        self.runtime=runtime*second
        self.dt=0.01*ms     		# simulation step length [ms]

	self.gEEA=gEEA*nS    
	self.gEEN=gEEN*nS   
	self.gEIA=gEIA*nS   
	self.gEIN=gEIN*nS 


       	self.gIE=gIE*nS 		# gaba conductances on pyramidal
        self.gII=gII*nS 		# gaba conductances on interneurons
        self.gextE=gextE*nS 		# external input condunctances mediated by AMPA on pyramidal cells (excitatory)
        self.gextI=gextI*nS 		# external input condanctances mediated by AMPA on interneurons (inihibitory)
        self.sigma= sigma		# E-to-E footprint in degrees
       	self.Jp=Jp 			# E-to-E footprint J_plus
	self.JpIE=JpIE 			# E-to-I, I->E footprint J_plus
	self.JpEI=JpEI 			# E-to-I, I->E footprint J_plus
	self.sigmaIE=sigmaIE 		# E-to-I, I->E footprint J_plus in degrees
	self.sigmaEI=sigmaEI
       	self.NE=NE
        self.NI=NI

   
    def plot(self,fd=None):
        
	t ='''gEEA= %s gEEN=%s
        gEIA=%s gEIN=%s gIE=%s gII=%s
        gextE=%s \t gextI=%s'''
	
	if fd: self.load = False
	else: fd=str(self.pid)+".model"
	
	if not self.load:
		f = file(fd, 'r')
	        #import pdb; pdb.set_trace()

		(spikesE,spikesI,sptimesE,sptimesI,counts,params)=load(f)
		self.spikesE=(spikesE,sptimesE)
	    	self.spikesI=(spikesI,sptimesI)
		self.counts=counts
		self.gEEA=params["gEEA"];self.gEEN=params["gEEN"]
		self.gEIA=params["gEIA"];self.gEIN=params["gEIN"]
		self.gIE=params["gIE"];self.gII=params["gII"]
		self.gextE=params["gextE"];self.gextI=params["gextI"]
		print params
		f.close()
		self.load=True

	bn=0.005
	# Excitatory neurons
	spiketimes=self.spikesE[1].values()
	NE=self.NE
	self.bump1 = spiketimes

	# colapse all neurons and sort
	self.bump1 = sorted(reduce(lambda x,y: x+list(y),self.bump1,[]))
	hist1 = histogram(self.bump1, arange(0,2.6,bn)); self.hist1=hist1

	# get only delay period -> from stim_off on
	i1=hist1[1] > self.stim_off
	i1=i1[1:] # remove head
	
	#ft1=rfft(hist1[0][i1])	
	
	subplot(231) # E raster plot
	raster_plot_spiketimes(self.spikesE[0])
	axhline(y=NE/2,color='red',ls='dashed') 			#bump2
	axvspan(self.stim_on/ms,self.stim_off/ms,color='gray',alpha=0.2) # stimulus
	ylim(0,self.NE)
	xlim(0,self.runtime/second*1000)

	subplot(232) # I raster plot
	raster_plot_spiketimes(self.spikesI[0])
	axhline(y=NE/2,color='red',ls='dashed') 			#bump2
	axvspan(self.stim_on/ms,self.stim_off/ms, color='gray',alpha=0.2) # stimulus
	ylim(0,self.NI)
	xlim(0,self.runtime/second*1000)

	subplot(233) # the mean activity in the end
	plot(counts/(self.runtime*0.25))
	xlim(0,self.NE)
	ylabel('firing rate (Hz)')
	xlabel('neuron')
	title('mean activity in last quarter')

	#import pdb; pdb.set_trace()
	subplot(234) # autocorrelogram
#	crossc=cross(ft1,ft1,bn)
#        deltas=linspace(-.2,.2,len(crossc))
#	plot(deltas,crossc)
#	axhline(y=0,color='black',ls='dashed')
	xlim(-0.2,0.2)
	title('autocorrelogram')

	subplot(235) # power spectrum
	psd(hist1[0],Fs=1/bn)
	xlim(0,100)
	title('power spectrum')

	suptitle(t % (self.gEEA, self.gEEN,self.gEIA,self.gEIN,self.gIE,self.gII,self.gextE,self.gextI))
	show()


    def run(self):
	seed(int(time.time()))
	defaultclock.dt = self.dt

        gEEA=self.gEEA
        gEEN=self.gEEN
        gEIA=self.gEIA
        gEIN=self.gEIN


        gIE=self.gIE
        gII=self.gII
        gextE=self.gextE
        gextI=self.gextI
        sigma=self.sigma
        Jp=self.Jp

        sigmaIE=self.sigmaIE
        JpIE=self.JpIE

        sigmaEI=self.sigmaEI
        JpEI=self.JpEI

        NE=self.NE
        NI=self.NI
        
        #these are intermediate calculations needed for the equations below
        N=NE+NI
        
	sig=sigma/360.0*NE
        fct=2.0*sum(exp(-0.5*(linspace(1,NE/2,NE/2)/sig)**2))/NE
        Jm=(1.0-Jp*fct)/(1.0-fct)

	sigIE=sigmaIE/360.0*NE
        fctIE=2.0*sum(exp(-0.5*(arange(0,NE/2,4)/sigIE)**2))/NI
        JmIE=(1.0-JpIE*fctIE)/(1.0-fctIE)

	sigEI=sigmaEI/360.0*NE
        fctEI=2.0*sum(exp(-0.5*(linspace(1,NE/2,NE/2)/sigEI)**2))/NE
        JmEI=(1.0-JpEI*fctEI)/(1.0-fctEI)

	gEEA=gEEA/gLeakE/NE*2048
        gEEN=0.635*gEEN/gLeakE/NE*2048  # factor 0.635 needed to compensate for the fact that NMDA synapses
        gEIA=gEIA/gLeakI/NE*2048   # do not saturate as done in Compte et al. 2000
        gEIN=0.635*gEIN/gLeakI/NE*2048
        gIE=gIE/gLeakE/NI*512
        gII=gII/gLeakI/NI*512
        gextE=gextE/gLeakE
        gextI=gextI/gLeakI

        #ring model connectivity
        def connEE(k):
          if abs(k)<=NE/2:
            value=(Jm+(Jp-Jm)*exp(-0.5*(k/sig)**2))
          else:
            value=(Jm+(Jp-Jm)*exp(-0.5*((abs(k)-NE)/sig)**2))
          return value

        def connIE(k):
          if abs(k)<=NE/2:
            value=(JmIE+(JpIE-JmIE)*exp(-0.5*(k/sigIE)**2))
          else:
            value=(JmIE+(JpIE-JmIE)*exp(-0.5*((abs(k)-NE)/sigIE)**2))
          return value

        def connEI(k):
          if abs(k)<=NE/2:
            value=(JmEI+(JpEI-JmEI)*exp(-0.5*(k/sigEI)**2))
          else:
            value=(JmEI+(JpEI-JmEI)*exp(-0.5*((abs(k)-NE)/sigEI)**2))
          return value

	#equations for each neuron. xpre and satura are shadow variables that track the saturated state
        # of a neuron's outgoing NMDA synapses, and this is used as a "synaptic depression" effect on x
        # (see modulation='satura' below) so as to mimic the NMDA saturation that is modeled in 
        # Compte et al., 2000
        eqsE = '''
        dV/dt = (-gea*V-gen*V/(1.0+exp(-V/Ven)/3.57)-gi*(V+70*mV)-(V-El))/(tau) : volt
        dgea/dt = -gea/(taua)           : 1
        dgen/dt = -gen/(taun)+x/(2*ms)   : 1
        dx/dt = -x/(taux)               : 1     
        dgi/dt = -gi/(taug)             : 1
        dsatura/dt= (1.0-satura)/(taun)-xpre*satura/(2*ms) :1
        dxpre/dt= -xpre/(taux)          : 1
        tau : second
        '''
        eqsI = '''
        dV/dt = (-gea*V-gen*V/(1.0+exp(-V/Ven)/3.57)-gi*(V+70*mV)-(V-El))/(tau) : volt
        dgea/dt = -gea/(taua)           : 1
        dgen/dt = -gen/(taun)+x/(2*ms)   : 1
        dx/dt = -x/(taux)               : 1     
        dgi/dt = -gi/(taug)             : 1
        tau : second
        '''
        external=PoissonGroup(N,rates=1800*Hz)
        extinputE=external.subgroup(NE)
        extinputI=external.subgroup(NI)

        rates=zeros(NE)*Hz
	if self.stim == 3:        
		rates[NE/6-10:NE/6+10]=ones(20)*30*Hz
		rates[3*NE/6-10:3*NE/6+10]=ones(20)*30*Hz
	        rates[5*NE/6-10:5*NE/6+10]=ones(20)*30*Hz
	if self.stim == 2:
		rates[NE/4-10:NE/4+10]=ones(20)*30*Hz
	        rates[3*NE/4-10:3*NE/4+10]=ones(20)*30*Hz
	if self.stim == 1:
		rates[NE/2-10:NE/2+10]=ones(20)*30*Hz


	# Create first Network (I and E)

        inputlayer=PoissonGroup(NE,rates=lambda t: (t>self.stim_on)*(t<self.stim_off)*rates)

        networkE=NeuronGroup(NE,model=eqsE,threshold=Vt,reset="V=Vr;xpre+=1", refractory=refE)
        networkI=NeuronGroup(NI,model=eqsI,threshold=Vt,reset=Vr, refractory=refI)
        networkE.tau=CmE/gLeakE
        networkI.tau=CmI/gLeakI
        networkE.V = Vt-2*mV + rand(NE) * 2*mV
        networkI.V = Vt-2*mV + rand(NI) * 2*mV

        extconnE=IdentityConnection(extinputE,networkE,'gea',weight=gextE)
        extconnI=IdentityConnection(extinputI,networkI,'gea',weight=gextI)

        inputmap=lambda i,j:exp(-0.5*((i-j)/sig)**2)*0.2
        feedforward=Connection(inputlayer,networkE,'gea',weight=inputmap)

        lateralmapA=lambda i,j:connEE(i-j)*gEEA
        lateralmapN=lambda i,j:connEE(i-j)*gEEN
        recurrentEEA=Connection(networkE, networkE, 'gea', weight=lateralmapA)
        recurrentEEN=Connection(networkE, networkE, 'x', weight=lateralmapN, modulation='satura')

        lateralmapAEI=lambda i,j:connEI(i-4*j)*gEIA
        lateralmapNEI=lambda i,j:connEI(i-4*j)*gEIN
        recurrentEIA=Connection(networkE, networkI, 'gea', weight=lateralmapAEI)
        recurrentEIN=Connection(networkE, networkI, 'x', weight=lateralmapNEI, modulation='satura')

	lateralmapIE=lambda i,j:connIE(4*i-j)*gIE
        recurrentIE=Connection(networkI, networkE, 'gi', weight=lateralmapIE)

        recurrentII=Connection(networkI, networkI, 'gi', weight=gII)


	# Monitors of activity
        spikesE=SpikeMonitor(networkE)    
	spikesI=SpikeMonitor(networkI)
        run(self.runtime*0.75,report="text")
	counts=SpikeCounter(networkE)
	run(self.runtime*0.25,report="text")
	f = file(str(self.pid)+".model", 'w')

	params={"gEEA":self.gEEA,"gEEN":self.gEEN,
		"gEIA":self.gEIA,"gEIN":self.gEIN,
		"gIE":self.gIE,"gII":self.gII,
		"gextE":self.gextE,"gextI":self.gextI,
		"sigma":self.sigma,"Jp":self.Jp,
		"sigmaIE":self.sigmaIE,"sigmaEI":self.sigmaEI,
		"JpIE":self.JpIE,"JpIE":self.JpIE,}

	todump=(spikesE.spikes,spikesI.spikes,spikesE.spiketimes,spikesI.spiketimes,counts.count,params)
	dump(todump, f, protocol=HIGHEST_PROTOCOL)
	f.close()
	
	print sum(rates)

