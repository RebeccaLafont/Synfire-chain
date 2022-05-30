#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:40:26 2022

@author: bk
"""


import matplotlib.pyplot as plt
import numpy as np

#constants
s = 1.0
ms = 0.001
debug = False

# simulation parameters
seed = 1
dt = 0.1*ms
duration  = 100 * ms
n_iterations = int(np.ceil(duration/dt)) 
t = np.linspace(0, duration, n_iterations)
in_number=30            #number of inhibitory neurons
tau_neuron  = 10*ms    
tau_synapse = 5*ms   
tau_inhibition= 1*ms #(????)
V_rest  = -60
V_reset = -55
V_spike = -50
layer_number = 10
layer_size = 15
neuron_nr1=layer_number*layer_size

#adaptation
a=-0.5#*nS
b=0.5#*nA
tau_k=100*ms
Wk=np.zeros((150,n_iterations))

#inhibition
 #not taking in any vairables ???
Vi=np.zeros((in_number,n_iterations))-60 #potential for inhibatory neurons
II=np.zeros((in_number, n_iterations)) #input to test V, V1, peaks = adapted_potential(weight)
II[:,:]=30
#for i in range(in_number):
    #II[i,40*i:40*(i+1)]=35        #input for all neurons at different times)
    
Wie=np.zeros((30,150)) #weight matrix between I and E
#excitatory
V1 = np.zeros((layer_number, layer_size,n_iterations))-60 
V = V1.reshape(layer_number*layer_size,n_iterations) 
I = np.zeros((layer_number*layer_size, n_iterations))
I[:15,:120] =30 #external current given only for the first layer, for 5ms
W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))

#weightI=3
 #a boolean array that confirms spike or not

#excitatory
def synaptic_weight1(layer_number,layer_size,weight):
  W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))
  for i in range(layer_number-1):
    W1[i+1,:,i,:]=weight
  neuron_nr1=layer_number*layer_size #number of neuron per layer
  W1=W1.reshape(neuron_nr1,neuron_nr1)#shape of layers
  return W1, neuron_nr1

def weightIE(in_number,weightI):
    for i in range(in_number):
            Wie[i,i*5:(i+1)*5]=weightI
    return Wie
def create_all_spikes1(V=V, dt=dt, threshold=V_spike):
  spikes_yn1=(V[:,1]>=threshold) #spike_yn1 is a vector 1*15
  last_spikes1=np.zeros(len(spikes_yn1)) #last_spikes1 is a vector 1*15
  for j in range (len(spikes_yn1)):
      if spikes_yn1[j]==1:
        last_spikes1[j]=dt
      else:
        last_spikes1[j]=last_spikes1[j]

  all_spikes1=np.column_stack((last_spikes1,last_spikes1)) #all_spikes_yn is a matrix (150*2)
  all_spikes1=all_spikes1.tolist()
  return all_spikes1, spikes_yn1, last_spikes1

def create_all_spikesI(VI, dt, V_spike):
    spikes_ynI=(VI[:,1]>=V_spike)
    last_spikesI=spikes_ynI*dt
    
    for j in range(len(spikes_ynI)):
        if spikes_ynI[j] == 1:#if there is a spike, find the time when it happened
            last_spikesI[j]=dt 
        else:
            last_spikesI[j]=0
            
    all_spikesI = np.column_stack((last_spikesI,last_spikesI))
    all_spikesI = all_spikesI.tolist() 
    return spikes_ynI, last_spikesI, all_spikesI


def postsyn_potential1(neuron_nr1, n, taus, all_spikes1, k, dt):
  E = np.zeros(neuron_nr1) #now neuron_nr is 150
  for m in range(neuron_nr1):
    e = 0

    f_times =np.unique(all_spikes1[m]) 
    
    for n in range(len(f_times)):
      et = np.exp(-((k+2)*dt-f_times[n])/taus) *np.sign(f_times[n])
      e=e+et 
    
    E[m] = e
  return E

def potentialI(weight):
    peaksI=[]
    spikes_ynI, last_spikesI, all_spikesI=create_all_spikesI(Vi, dt, V_spike)
    for k in range(n_iterations-100):
        derivV1=(-(Vi[:,k+1]-V_rest)+II[:,k+1])/tau_inhibition
        Vi[:,k+2]=Vi[:,k+1]+dt*derivV1
        for i in range(len(all_spikesI)):
          if spikes_ynI[i]==1:
              Vi[i][k+2]=V_rest
             
        spikes_ynI = (Vi[:,k+2]>=V_spike)
    
        A = np.argwhere(spikes_ynI == True)
        for j in range(len(spikes_ynI)):
          if spikes_ynI[j] == 1:
            last_spikesI[j] = (k+2)*dt
          else:
            last_spikesI[j] = last_spikesI[j]
                    
        for m in range(in_number):
          if last_spikesI[m] not in all_spikesI[m]:
            all_spikesI[m].append(last_spikesI[m])
        peaksI.append(A)
    return Vi, peaksI
#linking E and I : structed (3 IN per layer) 


def adapted_potential(weight,VI,weightI):
  all_spikes1, spikes_yn1, last_spikes1=create_all_spikes1()
  spikes_ynI, last_spikesI, all_spikesI=create_all_spikesI(Vi, dt, V_spike)
  VI,peaksI=potentialI(weight)
  peaks=[]
  for k in range (n_iterations-100): #for one k looking at matrix
      W1,neuron_nr1=synaptic_weight1(layer_number,layer_size,weight) #matrix 150*150
      E=postsyn_potential1(neuron_nr1,n_iterations,tau_synapse,all_spikes1,k,dt) #vector 1*150 ===> (2,)????
      I_syn=np.dot(W1,E) #vector 1*150?
      Wie=weightIE(in_number,weightI) #30*150
      
      for j in range(in_number-1): #so one I activates 5 E
          if spikes_ynI[j]==0: #but only if its active
              V[j*5:(j+1)*5,k+2]=V[j*5:(j+1)*5,k+1]+(-(V[j*5:(j+1)*5,k+1]-V_rest)+I_syn[j*5:(j+1)*5]+I[j*5:(j+1)*5,k+1]-Wk[j*5:(j+1)*5,k+1])*dt/tau_neuron #150*1
              Wk[j*5:(j+1)*5,k+2]=Wk[j*5:(j+1)*5,k+1]+(a*(V[j*5:(j+1)*5,k+1]-V_rest)-Wk[j*5:(j+1)*5,k+1])*dt/tau_k
          else:
              V[j*5:(j+1)*5,k+2]=V[j*5:(j+1)*5,k+1]+(-(V[j*5:(j+1)*5,k+1]-V_rest)+I_syn[j*5:(j+1)*5]+I[j*5:(j+1)*5,k+1]-Wk[j*5:(j+1)*5,k+1]-Wie[j,j*5:(j+1)*5])*dt/tau_neuron #150*1
              Wk[j*5:(j+1)*5,k+2]=Wk[j*5:(j+1)*5,k+1]+(a*(V[j*5:(j+1)*5,k+1]-V_rest)-Wk[j*5:(j+1)*5,k+1])*dt/tau_k
     
      #reset if passes threshold
      for i in range(len(all_spikes1)):
        if spikes_yn1[i]==1:
          V[i,k+2]=V_reset
          Wk[i,k+2]=Wk[i,k+2]+b #Wk[i,k+1]+(a*(V[i,k+1]-V_rest) - Wk[i,k+1])*dt/tau_k-0.05
          
        elif spikes_yn1[i] == 0:
          V[i,k+2] = V[i,k+2]
      #link to In
      
        
      spikes_yn1=(V[:,k+2]>=V_spike)
      spikes_ynI=(VI[:,k+2]>=V_spike)
      #record spikes
      B=np.argwhere(spikes_yn1==True)
      for j in range(len(spikes_yn1)):
        if spikes_yn1[j]==1:
          last_spikes1[j]=(k+2)*dt
        else:
          last_spikes1[j]=last_spikes1[j]
      for m in range(neuron_nr1):
        if last_spikes1[m] not in all_spikes1[m]:
          all_spikes1[m].append(last_spikes1[m])
      peaks.append(B)
  return V, V1, peaks
#inhibition


  

    
    
#PLOT
def PLOT_all(weight,weightI):
    V, V1, peaks = adapted_potential(weight,Vi,weightI)
     
    ax = plt.subplot(3,1,1)
    plt.title("Evolution of potential in excitatory population w="+str(weight) )
    for n in range(neuron_nr1):
        ax.plot(t, V[n,:], color= "red", alpha = 0.25, linewidth = 0.5)
    ax.set_ylabel("Potential")
    plt.xlim(0,0.10)
    
    ax = plt.subplot(3,1,2) #Raster
    plt.title("Spiking times for E neurons")
    plt.ylim(0,neuron_nr1)
    plt.xlim(0,0.06)
    ax.set_ylabel("Neuron Index")
    spikes = []
    for i in range(len(peaks)):
      for j in range(len(peaks[i])):
        plt.plot(t[i],peaks[i][j]
                 , "|", color = "darkred")
        
    ax=plt.subplot(3,1,3)
    plt.title("Evolution of potential in inhibitory population wI="+str(weightI))
    for n in range(in_number):
        ax.plot(t,Vi[n,:], color= "red", alpha= 0.25, linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Potential")
    
    plt.tight_layout()

def PLOT_E(weight,weightI):
    V, V1, peaks = adapted_potential(weight,Vi,weightI)
    fig = plt.figure(figsize = (8,10))
    ax=plt.subplot(2,1,1)
    plt.title("Evolution of potential in Excitatory population  - weight="+str(weight) )
    for n in range(neuron_nr1):
        ax.plot(t, V[n,:], color= "red", alpha = 0.25, linewidth = 0.25)
    ax.plot(t, V[0,:], color= "darkred", alpha = 1, linewidth = 1.5)
    ax.set_ylabel("Potential")
    plt.xlim(0,0.06)
    
    ax=plt.subplot(2,1,2)
    plt.title("Spiking times for E neurons")
    plt.ylim(0,neuron_nr1)
    plt.xlim(0,0.06)
    ax.set_ylabel("Neuron Index")
    ax.set_xlabel("Time (s)")
    spikes = []
    for i in range(len(peaks)):
      for j in range(len(peaks[i])):
        plt.plot(t[i],peaks[i][j]
                 , "|", color = "darkred")
    plt.tight_layout()