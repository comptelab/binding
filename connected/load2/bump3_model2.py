import numpy as np
from scipy import fftpack
import scipy.io
import pylab as pl
import scipy.signal as signal
from brian import *
from numpy.fft import rfft,irfft
from numpy.random import seed
from copy import deepcopy
from multiprocessing import Process,Queue,Pipe
from ctypes import *
from pickle import *
from threading import *
from bisect import insort
import socket
from time import time

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

B=0
A=1
D=2
C=3

#stim1=3*1024/4
#stim2=1024/2
#stim3=1024/4

# three
#stim1=850
#stim2=500
#stim3=150



def cross(bump1,bump2,bn):
        spec=bump1*conj(bump2)
        l=irfft(spec)[-int(0.2/bn)-1:]
        r=irfft(spec)[:int(0.2/bn+1)]
        crossc=list(l)+list(r)
        crossc-=mean(crossc)
        return crossc



class Model(Process):
    # Sets default parameters (no AMPA, no oscilations) if none given

    def __init__(self,on1=500,on2=725,off1=750,off2=750,runtime=3.5,
		giEEA=0,giEEN=0,giEIA=0,giEIN=0,
		gEEA=0,gEEN=0.484,gEIA=0,gEIN=0.379,
                gIE=3.643,gII=2.896,gextE=6.6,gextI=5.8,
		sigma= 9,sigmaIE=32.4,sigmaEI=32.4,
		Jp=5.7,JpIE=1.4,JpEI=1.4,
		NE=2048,NI=512,stim=1,stims=[1500,1000,500],extra=False):

        Process.__init__(self)
	self.load=False
	self.spikesI=None
	self.spikesE=None
	self.counts=None
	self.phase=None
	self.extra=extra
        self.stim1_on=on1*ms
        self.stim1_off=off1*ms
        self.stim2_on=on2*ms
        self.stim2_off=off2*ms
	self.stimulus=stim 		# set to 1 to turn on and to 0 to turn off external stimulus
	self.stim1=stims[0]
	self.stim2=stims[1]
	self.stim3=stims[2]
        self.runtime=runtime*second
        self.dt=0.01*ms     		# simulation step length [ms]

	self.gEEA=gEEA*nS    
	self.gEEN=gEEN*nS   
	self.gEIA=gEIA*nS   
	self.gEIN=gEIN*nS 


	self.giEEA=giEEA*nS
	self.giEEN=giEEN*nS
	self.giEIN=giEIN*nS
	self.giEIA=giEIA*nS


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

    def convert_to_matlab (self,fd):
	f = file(fd,'r')
	(spikesE1,spikesI1,sptimesE1,sptimesI1,counts1,spikesE2,spikesI2,sptimesE2,sptimesI2,counts2,params)=load(f)
	print len(spikesE1)
	#scipy.io.savemat(fd+'.mat',{'sptimesE1':sptimesE1})
	#scipy.io.savemat(fd+'.mat',{'sptimesE1': sptimesE1,  'sptimesE2': sptimesE2,'sptimesI1': sptimesI1,  'sptimesI2': sptimesI2 })
        scipy.io.savemat(fd+'.mat',{'spikesE1':spikesE1, 'spikesE2':spikesE2},do_compression=True)
    def plot(self,fd=None,plotit=True):
        
	t ='''giEEA= %s giEEN=%s
        giEIA=%s giEIN=%s gEIA=%s gEEN=%s
	gextE=%s gextI=%s'''
	
	if fd: self.load = False
	else: fd="%s_%s.model" % (self.pid,socket.gethostname())
	
	if not self.load:
		f = file(fd, 'r')
		self.f=f
		(spikesE1,spikesI1,sptimesE1,sptimesI1,counts1,spikesE2,spikesI2,sptimesE2,sptimesI2,counts2,params)=load(f)
		self.spikesE1=(spikesE1,sptimesE1)
	    	self.spikesI1=(spikesI1,sptimesI1)
		self.counts1=counts1

		self.spikesE2=(spikesE2,sptimesE2)
	    	self.spikesI2=(spikesI2,sptimesI2)
		self.counts2=counts2

		self.gEEA=params["gEEA"];self.gEEN=params["gEEN"]
		self.gEIA=params["gEIA"];self.gEIN=params["gEIN"]
		self.gIE=params["gIE"];self.gII=params["gII"]
		self.giEEA=params["giEEA"];self.giEIA=params["giEIA"]
		self.giEEN=params["giEEN"];self.giEIN=params["giEIN"]
		self.gextE=params["gextE"];self.gextI=params["gextI"]
		self.stim1=params["stims"][0]
		self.stim2=params["stims"][1]
		self.stim3=params["stims"][2]
		self.extra=params["extra"]
		f.close()
		self.load=True

	bn=0.005

	#### NETWORK 1
	# Excitatory neurons
	spiketimes1=self.spikesE1[1].values()
	NE=self.NE
        self.bump11 = spiketimes1[2*NE/3:NE]
        self.bump21 = spiketimes1[NE/3:2*NE/3]
        self.bump31 = spiketimes1[0:NE/3]
	
	# colapse all neurons and sort
        self.bump11 = sorted(reduce(lambda x,y: x+list(y),self.bump11,[]))
        self.bump21 = sorted(reduce(lambda x,y: x+list(y),self.bump21,[]))
        self.bump31 = sorted(reduce(lambda x,y: x+list(y),self.bump31,[]))
        hist11 = histogram(self.bump11, arange(0,self.runtime/second,bn)); 
        hist21 = histogram(self.bump21, arange(0,self.runtime/second,bn)); 
        hist31 = histogram(self.bump31, arange(0,self.runtime/second,bn)); 

        # get only delay period -> from stim_off on
        i11=hist11[1] > self.stim1_off
        i11=i11[1:] # remove head
        i11_b=hist11[1] > self.stim1_off
        i11_b=i11_b[1:] # remove head
        i21=hist21[1] > self.stim1_off
        i21=i21[1:]
        i31=hist31[1] > self.stim1_off
        i31=i31[1:]

        ft11=rfft(hist11[0][i11])
        ft21=rfft(hist21[0][i21])
        ft31=rfft(hist31[0][i31])
	#import pdb; pdb.set_trace()

	#### NETWORK 2
	# Excitatory neurons
	spiketimes2=self.spikesE2[1].values()
	NE=self.NE
        self.bump12 = spiketimes2[2*NE/3:NE]
        self.bump22 = spiketimes2[NE/3:2*NE/3]
        self.bump32 = spiketimes2[0:NE/3]

	# colapse all neurons and sort
        self.bump12 = sorted(reduce(lambda x,y: x+list(y),self.bump12,[]))
        self.bump22 = sorted(reduce(lambda x,y: x+list(y),self.bump22,[]))
        self.bump32 = sorted(reduce(lambda x,y: x+list(y),self.bump32,[]))
        hist12 = histogram(self.bump12, arange(0,self.runtime/second,bn)); self.hist12=hist12
        hist22 = histogram(self.bump22, arange(0,self.runtime/second,bn)); self.hist22=hist22
        hist32 = histogram(self.bump32, arange(0,self.runtime/second,bn)); self.hist32=hist32

        # get only delay period -> from stim_off on
        i12=hist12[1] > self.stim1_off
        i12=i12[1:] # remove head
        i12_b=hist12[1] > self.stim1_off
        i12_b=i12_b[1:] # remove head
        i22=hist22[1] > self.stim1_off
        i22=i22[1:]
        i32=hist32[1] > self.stim1_off
        i32=i32[1:]

        ft12=rfft(hist12[0][i12])
        ft22=rfft(hist22[0][i22])
        ft32=rfft(hist32[0][i32])

	crosscs=[]
	for bump1 in [ft11,ft21,ft31,ft12,ft22,ft32]:
		line=[]
		for bump2 in [ft11,ft21,ft31,ft12,ft22,ft32]:
			line.append(cross(bump1,bump2,bn))
		crosscs.append(line)
	self.crosscs=crosscs

	if plotit: 
		subplot(231) # E1 raster plot
		title("network 1")
		raster_plot_spiketimes(self.spikesE1[0])
		axhline(y=self.stim1,color='red',ls='dashed') 			#bump1
		axhline(y=self.stim2,color='red',ls='dashed') 			#bump2
		axhline(y=self.stim3,color='red',ls='dashed') 			#bump3
		axvspan(self.stim1_on/ms,self.stim1_off/ms,color='gray',alpha=0.2) # stimulus
		axvspan(self.stim2_on/ms,self.stim2_off/ms,color='green',alpha=0.2) # stimulus
		axvspan(self.runtime*0.75/ms,self.runtime/ms,color='gray',alpha=0.2) # stimulus
		ylim(0,self.NE)
		xlim(0,self.runtime/second*1000)

		subplot(232) # E2 raster plot
		title("network 2")
		raster_plot_spiketimes(self.spikesE2[0])
		axhline(y=self.stim1,color='red',ls='dashed') 			#bump1
		axhline(y=self.stim2,color='red',ls='dashed') 			#bump2
		axhline(y=self.stim3,color='red',ls='dashed') 			#bump3
		axvspan(self.stim1_on/ms,self.stim1_off/ms, color='gray',alpha=0.2) # stimulus
		axvspan(self.stim2_on/ms,self.stim2_off/ms, color='green',alpha=0.2) # stimulus
		axvspan(self.runtime*0.75/ms,self.runtime/ms,color='red',alpha=0.2) # stimulus
		ylim(0,self.NE)
		xlim(0,self.runtime/second*1000)

		subplot(233) # the mean activity in the end
		plot(counts1/(self.runtime*0.25),label="network 1")
		plot(counts2/(self.runtime*0.25),label="network 2")
		xlim(0,self.NE)
		legend()
		ylabel('firing rate (Hz)')
		xlabel('neuron')
		title('mean activity in last quarter')




		deltas=linspace(-.2,.2,len(crosscs[0][0]))

		subplot(234) # cross spectrum
	#	csd(hist11[0],hist12[0],Fs=1/bn)
	#	psd(hist11[0],Fs=1/bn)
		psd(hist12[0],Fs=1/bn)
		xlim(0,100)
		legend(framealpha=0.5)
		title('power spectrum')
		xlim(0,50)
	
		subplot(235)
		
		#import pdb; pdb.set_trace()
		w=20
		w2=1
		h11=hist11[0][i11]
		h11_b=hist11[0][i11_b]
		h31=hist31[0][i31]
		
		h12=hist12[0][i12]		
		h12_b=hist12[0][i12_b]		
		h32=hist32[0][i32]

		cs=[]
		cs2=[]
		csin=[]
		for i in range(0,len(h11),w2):
			framein1=h11[i:i+w]
			framein2=h31[i:i+w]
			csin+=[corrcoef(framein1,framein2)[0][1]]

			frame1=h11[i:i+w]
			frame2=h12[i:i+w]
			cs+=[corrcoef(frame1,frame2)[0][1]]
	
			frame11=h11[i:i+w]
			frame22=h32[i:i+w]
			cs2+=[corrcoef(frame11,frame22)[0][1]]

		title("internetwork bump correlation")
		plot(range(0,len(h11),w2),cs,"b",label="initially bound")
		plot(range(0,len(h11),w2),cs2,"r",label="initially unbound")
		legend()
	#	plot(range(0,len(h11),w2),csin,"g")
			
		
		
		suptitle(t % (self.giEEA, self.giEEN,self.giEIA,self.giEIN, self.gEIA,self.gEEN,self.gextE,self.gextI))
		print params
		show()


	def cor(a,b): return corrcoef(a,b)[0][1]

	n=len(crosscs[B][C])/2
	print n
	self.BD=crosscs[B][D][n-1:n+2]
	self.BC=crosscs[B][C][n-1:n+2]
	self.AD=crosscs[A][D][n-1:n+2]
	self.AC=crosscs[A][C][n-1:n+2]
	self.BA=crosscs[B][A][n-1:n+2]
	self.DC=crosscs[D][C][n-1:n+2]

    def run(self):
	seed(mod(id(self),2**32))
	defaultclock.dt = self.dt
	
	giEEA=self.giEEA
	giEEN=self.giEEN
	giEIN=self.giEIN
	giEIA=self.giEIA


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

	giEEA=giEEA/gLeakE/NE*2048
	giEIA=giEIA/gLeakI/NE*2048

#        giEIN=0.635*giEIN/gLeakI/NE*2048
#        giEEN=0.635*giEEN/gLeakI/NE*2048
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

	
 
	probe_stim = self.stim3

	rates11=zeros(NE)*Hz
        rates21=zeros(NE)*Hz
	rates_probe = zeros(NE)*Hz 

        rates_probe[self.stim3-10:self.stim3+10]=ones(20)#*10*Hz

        rates11[self.stim1-10:self.stim1+10]=ones(20)#*10*Hz
        #rates[self.stim2-10:self.stim2+10]=ones(20)*30*Hz
        rates11[self.stim3-10:self.stim3+10]=ones(20)#*10*Hz
	rates21[self.stim1-10:self.stim1+10]=ones(20)#*5*Hz
	
        rates12=zeros(NE)*Hz
        rates22=zeros(NE)*Hz
        
        rates12[self.stim1-10:self.stim1+10]=ones(20)#*10*Hz
        #rates22[self.stim2-10:self.stim2+10]=ones(20)#*10*Hz
        rates12[self.stim3-10:self.stim3+10]=ones(20)#*30*Hz
	rates22[self.stim1-10:self.stim1+10]=ones(20)#*5*Hz

	def stims(time,rates1,rates2,probe=0):
		s1=(time >self.stim1_on)*(time<self.stim1_off)*rates1*20*Hz
		s2=(time >self.stim2_on)*(time<self.stim2_off)*rates2*150*Hz
		s3=probe*(time >self.runtime*0.75)*rates_probe*5*Hz

		return s1+s2+s3
		

	# Create first Network (I and E)

        external1=PoissonGroup(N,rates=1800*Hz)
        extinputE1=external1.subgroup(NE)
        extinputI1=external1.subgroup(NI)

        #inputlayer1=PoissonGroup(NE,rates=lambda t: (t>self.stim_on)*(t<self.stim_off)*rates)
        inputlayer1=PoissonGroup(NE,rates=lambda t: stims(t,rates11,rates21,1))

        networkE1=NeuronGroup(NE,model=eqsE,threshold=Vt,reset="V=Vr;xpre+=1", refractory=refE)
        networkI1=NeuronGroup(NI,model=eqsI,threshold=Vt,reset=Vr, refractory=refI)
        networkE1.tau=CmE/gLeakE
        networkI1.tau=CmI/gLeakI
        networkE1.V = Vt-2*mV + rand(NE) * 2*mV
        networkI1.V = Vt-2*mV + rand(NI) * 2*mV

        extconnE1=IdentityConnection(extinputE1,networkE1,'gea',weight=gextE)
        extconnI1=IdentityConnection(extinputI1,networkI1,'gea',weight=gextI)

        inputmap=lambda i,j:exp(-0.5*((i-j)/sig)**2)*0.2*self.stimulus
        feedforward1=Connection(inputlayer1,networkE1,'gea',weight=inputmap)

        lateralmapA=lambda i,j:connEE(i-j)*gEEA
        lateralmapN=lambda i,j:connEE(i-j)*gEEN
        recurrentEEA1=Connection(networkE1, networkE1, 'gea', weight=lateralmapA)
        recurrentEEN1=Connection(networkE1, networkE1, 'x', weight=lateralmapN, modulation='satura')

        lateralmapAEI=lambda i,j:connEI(i-4*j)*gEIA
        lateralmapNEI=lambda i,j:connEI(i-4*j)*gEIN
        recurrentEIA1=Connection(networkE1, networkI1, 'gea', weight=lateralmapAEI)
        recurrentEIN1=Connection(networkE1, networkI1, 'x', weight=lateralmapNEI, modulation='satura')

	lateralmapIE=lambda i,j:connIE(4*i-j)*gIE
        recurrentIE1=Connection(networkI1, networkE1, 'gi', weight=lateralmapIE)

        recurrentII1=Connection(networkI1, networkI1, 'gi', weight=gII)

	# Create second Network (I and E)
        external2=PoissonGroup(N,rates=1800*Hz)
        extinputE2=external2.subgroup(NE)
        extinputI2=external2.subgroup(NI)

        #inputlayer2=PoissonGroup(NE,rates=lambda t: (t>self.stim_on)*(t<self.stim_off)*rates)
        inputlayer2=PoissonGroup(NE,rates=lambda t: stims(t,rates12,rates22))

        networkE2=NeuronGroup(NE,model=eqsE,threshold=Vt,reset="V=Vr;xpre+=1", refractory=refE)
        networkI2=NeuronGroup(NI,model=eqsI,threshold=Vt,reset=Vr, refractory=refI)
        networkE2.tau=CmE/gLeakE
        networkI2.tau=CmI/gLeakI
        networkE2.V = Vt-2*mV + rand(NE) * 2*mV
        networkI2.V = Vt-2*mV + rand(NI) * 2*mV

        extconnE2=IdentityConnection(extinputE2,networkE2,'gea',weight=gextE)
        extconnI2=IdentityConnection(extinputI2,networkI2,'gea',weight=gextI)

        feedforward2=Connection(inputlayer2,networkE2,'gea',weight=inputmap)

        recurrentEEA2=Connection(networkE2, networkE2, 'gea', weight=lateralmapA)
        recurrentEEN2=Connection(networkE2, networkE2, 'x', weight=lateralmapN, modulation='satura')

        recurrentEIA2=Connection(networkE2, networkI2, 'gea', weight=lateralmapAEI)
        recurrentEIN2=Connection(networkE2, networkI2, 'x', weight=lateralmapNEI, modulation='satura')

        recurrentIE2=Connection(networkI2, networkE2, 'gi', weight=lateralmapIE)
        recurrentII2=Connection(networkI2, networkI2, 'gi', weight=gII)


	# Connect both Netowrks
	# N1->N2 & N2->N1
	intraEEA1=Connection(networkE1,networkE2,'gea',weight=giEEA)
#	intraEEN1=Connection(networkE1,networkE2,'x',weight=giEEN)
	intraEEA2=Connection(networkE2,networkE1,'gea',weight=giEEA)
#	intraEEN2=Connection(networkE2,networkE1,'x',weight=giEEN)

	intraEIA1=Connection(networkE1,networkI2,'gea',weight=giEIA)
#	intraEIN1=Connection(networkE1,networkI2,'x',weight=giEIN)
	intraEIA2=Connection(networkE2,networkI1,'gea',weight=giEIA)
#	intraEIN2=Connection(networkE2,networkI1,'x',weight=giEIN)

	# Monitors of activity
        spikesE1=SpikeMonitor(networkE1)    
	spikesI1=SpikeMonitor(networkI1)
        spikesE2=SpikeMonitor(networkE2)    
	spikesI2=SpikeMonitor(networkI2)

        run(self.runtime*0.75,report="text")
	counts1=SpikeCounter(networkE1)
	counts2=SpikeCounter(networkE2)
	run(self.runtime*0.25,report="text")
#	f = file(str(self.pid)+".model", 'w')
	f= file("%s_%s.model" % (self.pid,socket.gethostname()),"w")

	params={"gEEA":self.gEEA,"gEEN":self.gEEN,
		"gEIA":self.gEIA,"gEIN":self.gEIN,
		"gIE":self.gIE,"gII":self.gII,
		"gextE":self.gextE,"gextI":self.gextI,
		"giEEA":self.giEEA,"giEIA":self.giEIA,
		"giEEN":self.giEEN,"giEIN":self.giEIN,
		"sigma":self.sigma,"Jp":self.Jp,
		"sigmaIE":self.sigmaIE,"sigmaEI":self.sigmaEI,
		"JpIE":self.JpIE,"JpIE":self.JpIE,"extra":self.extra,"stims":[self.stim1,self.stim2,self.stim3]}

	todump=(spikesE1.spikes,spikesI1.spikes,spikesE1.spiketimes,spikesI1.spiketimes,counts1.count,spikesE2.spikes,spikesI2.spikes,spikesE2.spiketimes,spikesI2.spiketimes,counts2.count,params)
	print len(todump)
	dump(todump, f, protocol=HIGHEST_PROTOCOL)
	f.close()

'''
		subplot(234) 
		for i in range(3):
			for j in range(i+1,3):
				plot(deltas,crosscs[i][j])
		axhline(y=0,color='black',ls='dashed')
		xlim(-0.2,0.2)
		title('crosscorrelogram within')

		subplot(234)
		for i in range(3):
			plot(deltas,crosscs[B][i+3])
		axhline(y=0,color='black',ls='dashed')
		xlim(-0.2,0.2)
		title('crosscorrelogram between from TOP')

		subplot(235)
                for i in range(3):
                        plot(deltas,crosscs[1][i+3])
                axhline(y=0,color='black',ls='dashed')
                xlim(-0.2,0.2)
		title('crosscorrelogram between from MIDLE')

		subplot(236)
                for i in range(3):
                        plot(deltas,crosscs[2][i+3])
                axhline(y=0,color='black',ls='dashed')
                xlim(-0.2,0.2)
		title('crosscorrelogram between from BOTTOM')
'''


	
'''
	# begin and end of bumps
	begins=[]
	ends=[]
	for bump in [hist11,hist31,hist12,hist32]:
		ib=array(bump[1]>self.stim_off)*array(bump[1] < self.stim_off+(self.runtime-self.stim_off)/3)
		ie=array(bump[1] > self.runtime - (self.runtime-self.stim_off)/3)
		ftb=rfft(bump[0][ib[1:]])
		fte=rfft(bump[0][ie[1:]])
		begins.append(ftb)
		ends.append(fte)
		
	
	swaps=[]
	for b in begins:
		i=[]
		for e in begins:
			spec=b*conj(e)
			l=irfft(spec)[-0.2/bn-1:]
			r=irfft(spec)[:0.2/bn+1]
			crossc=list(l)+list(r)
			crossc-=mean(crossc)
			i.append(crossc[len(l)])
		swaps.append(i)
	self.swapsb=array(swaps)
		
	swaps=[]
	for b in ends:
		i=[]
		for e in ends:
			spec=b*conj(e)
			l=irfft(spec)[-0.2/bn-1:]
			r=irfft(spec)[:0.2/bn+1]
			crossc=list(l)+list(r)
			crossc-=mean(crossc)
			i.append(crossc[len(l)])
		swaps.append(i)
	self.swapse=array(swaps)
'''

