#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:43:14 2022

@author: bk
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

#SYNFIRE CHAIN
#constants
s = second = 1.0
ms = millisecond = 0.001
debug = False

# simulation parameters
seed = 1
dt = 0.1*ms
duration1  = 100 * ms
n_iterations = int(np.ceil(duration1/dt)) 

layer_number = 10
layer_size = 15
tau_synapse = 5*ms        #tau of the synapse
tau_neuron  = 10*ms       #tau of the integrate and fire neurons
V_rest  = -60
V_reset = -55
V_spike = -50
# building the model
V1 = np.zeros((layer_number, layer_size,n_iterations))-60 
#why are we reshaping? Any ideas?
V = V1.reshape(layer_number*layer_size,n_iterations) 
t1 = np.linspace(0, duration1, n_iterations)
I = np.zeros((layer_number*layer_size, n_iterations))
I[:15,:120] =30 #external current given only for the first layer, for 5ms
W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))
noise_mean = 0
noise_std = 2
noise_scale = np.sqrt(10*ms)

noise = np.random.normal(noise_mean, noise_std, n_iterations)
#plt.plot(noise*noise_scale)
#plt.plot(noise)

#adaptation parameters
nS=10**(-9)
nA=10**(-9)
a=-0.5#*nS
b=0.5#*nA
tau_k=100*ms
Wk=np.zeros((150,n_iterations))


def synaptic_weight1(layer_number,layer_size,weight):
  W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))
  for i in range(layer_number-1):
    W1[i+1,:,i,:]=weight
  neuron_nr1=layer_number*layer_size #number of neuron per layer
  W1=W1.reshape(neuron_nr1,neuron_nr1)#shape of layers
  return W1, neuron_nr1

W1,neuron_nr1=synaptic_weight1(layer_number,layer_size,3)


#plt.imshow(W)

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

def adapted_potential(weight):
  all_spikes1, spikes_yn1, last_spikes1=create_all_spikes1()
  peaks=[]
  for k in range (n_iterations-100): #for one k looking at matrix
      W1,neuron_nr1=synaptic_weight1(layer_number,layer_size,weight) #matrix 150*150
      E=postsyn_potential1(neuron_nr1,n_iterations,tau_synapse,all_spikes1,k,dt) #vector 1*150 ===> (2,)????
      I_syn=np.dot(W1,E) #vector 1*150?
      
      #print(I_syn)
      noise=np.random.normal(noise_mean,noise_std,neuron_nr1)
      #wk=0
      
      #for x in range(layer_number):
           #wk=wk+Wk[x*15,k+1]
      V[:,k+2]=V[:,k+1]+(-(V[:,k+1]-V_rest)+I_syn+I[:,k+1]-Wk[:,k+1]+noise_scale*noise)*dt/tau_neuron
      Wk[:,k+2]=Wk[:,k+1]+(a*(V[:,k+1]-V_rest)-Wk[:,k+1])*dt/tau_k
      #print(wk)
      #reset if passes threshold
      for i in range(len(all_spikes1)):
        if spikes_yn1[i]==1:
          V[i,k+2]=V_reset
          Wk[i,k+2]=Wk[i,k+2]+b #Wk[i,k+1]+(a*(V[i,k+1]-V_rest) - Wk[i,k+1])*dt/tau_k-0.05
          
        elif spikes_yn1[i] == 0:
          V[i,k+2] = V[i,k+2]
          
      
      spikes_yn1=(V[:,k+2]>=V_spike)
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

def plot_SYNFIRE(weight):
  V, V1, peaks = adapted_potential(weight)
  fig = plt.figure(figsize = (8,10))
  ax = plt.subplot(2,1,1)
  plt.title("Synaptic weight = " + str(weight) )
  for n in range(layer_number):
      ax.plot(t1, V1[n,0,:], color= "red", alpha = 0.25, linewidth = 0.5)
  ax.plot(t1, V1[4,0,:], color= "darkred", alpha = 1, linewidth = 1.5)
  plt.xlim(0,0.045)
  ax.set_ylabel("Potential")
  

  
  ax = plt.subplot(2,1,2) #Raster
  plt.title("Spiking times for neurons")
  plt.ylim(0,layer_size*layer_number)
  plt.xlim(0,0.045)
  ax.set_xlabel("Time (ms)")
  ax.set_ylabel("Neuron Index")
  
  spikes = []
  for i in range(len(peaks)):
    for j in range(len(peaks[i])):
      plt.plot(t1[i],peaks[i][j], "|", color = "darkred")

  plt.tight_layout()
  
  for i in tqdm(range(10)):
      sleep(0.2)