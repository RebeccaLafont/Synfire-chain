#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:07:04 2022

@author: bk
"""
import numpy as np
import matplotlib.pyplot as plt

#synfirechain with spike 4 times hardcoded

#constants
s = second = 1.0
ms = millisecond = 0.001
debug = False

# simulation parameters
seed = 1
dt = 0.01*ms
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
V1 = np.zeros((layer_number, layer_size,n_iterations))-60 #matrix of matrix
V = V1.reshape(layer_number*layer_size,n_iterations) #matrix
t1 = np.linspace(0, duration1, n_iterations)
I = np.zeros((layer_number*layer_size, n_iterations))
I[:15,:50] = 30  #external current given only for the first layer, for 5ms

W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))
#noise
noise_mean = 0
noise_std = 10
noise_scale = np.sqrt(10*ms)
noise = np.random.normal(noise_mean, noise_std, n_iterations)

'''Creating a weight matrix, where the layer are all connected by the same weight,
 except between the 4th and 5th parameter which is given as a parameter'''
def synaptic_weight1(layer_number,layer_size,weight):
  W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))
  for i in range(layer_number-1):
      W1[i+1,:,i,:]=1.5
  W1[0,:,-1,:] = 0
  neuron_nr1=layer_number*layer_size #number of neuron per layer
  W1=W1.reshape(neuron_nr1,neuron_nr1)#shape of layers
  W1[60:75,45:60]=weight #weight as parameter between the 4th and 5th layer
  return W1, neuron_nr1

W2,neuron_nr1=synaptic_weight1(layer_number,layer_size,3)

''' Create a function recording if there is a spike and the corresponding time
for the first timestep'''
def create_all_spikes1(V=V, dt=dt, V_spike=V_spike):
  spikes_yn1=(V[:,1]>=V_spike) #spike_yn1 is a vector 1*150
  last_spikes1=np.zeros(len(spikes_yn1)) #last_spikes1 is a vector 1*150
  for j in range (len(spikes_yn1)):
      if spikes_yn1[j]==1:
        last_spikes1[j]=dt
  all_spikes1=np.column_stack((last_spikes1,last_spikes1)) #all_spikes_yn is a matrix (150*2)
  all_spikes1=all_spikes1.tolist()
  return all_spikes1, spikes_yn1, last_spikes1

V[:15, (0,202,402,602)] = -30
V[:15,:50]=I[:15,:50]/tau_neuron*dt+V_rest

''' Builing  the final potential matrix
Create a higher value for artifical spike and 
lower for artificial rest to be able to differenciate 
them from the others'''
def weight_related_result1(weight):
  all_spikes1, spikes_yn1, last_spikes1=create_all_spikes1()
  peaks=[]
  for k in range (9900): #for each time step
      W1,neuron_nr1=synaptic_weight1(layer_number,layer_size,weight) #matrix 150*150
      #postsynaptic potential
      E = np.zeros(neuron_nr1) #now neuron_nr is 150
      for m in range(neuron_nr1):
        e = 0
        f_times =np.unique(all_spikes1[m]) 
        for n in range(len(f_times)):
          et = (np.exp(-((k+2)*dt-f_times[n])/tau_synapse)) *np.sign(f_times[n])
          e=e+et 
        E[m] = e
      I_syn=np.dot(W1,E) 
      #noise
      noise=np.random.normal(noise_mean,noise_std,neuron_nr1)
      #Euler's equation
      derivV1=(-(V[:,k+1]-V_rest)+I_syn+I[:,k+1]+(noise_scale*noise))/tau_neuron
     
      for i in range(neuron_nr1):
          if V[i,k+2]<-40 and V[i,k+2]>-80:   
              V[i,k+2]=V[i,k+1]+dt*derivV1[i] #apply equation when between artificial values
          elif V[i,k+2]==-80:
              V[i,k+2]=V_reset
          else:
              V[i,k+2]=V[i,k+2]
              
      
              
      for i in range(len(spikes_yn1)):
          if spikes_yn1[i]==1:
              if V[i,k+1]<-40: #if real spike
                  try:
                      V[i,k+2]=V_reset #reset at next time step
                      V[i,(k+3):(k+1003)]=-80  #between artificial spikes stay at rest (will go to reset at next time step, and equation at the one just after)
                      V[i,(k+202,k+402,k+602)]=-40 #create artificial spikes
                      
                  except IndexError:
                      pass
              elif V[i,k+1]>=-40:
                   V[i,k+2]=V_reset #if artificial spike, act like real spike and reset
          elif spikes_yn1[i]==0:
              V[i,k+2]=V[i,k+2]
         
      
      spikes_yn1=(V[:,k+2]>=V_spike) #spikes for this timestep
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

'''This function plots the potential evolution with the first layer in darkred,
 and creates a raster plot of the spike times '''
def plot_SYNFIRE1(weight):
  V, V1, peaks = weight_related_result1(weight)
  fig = plt.figure(figsize = (8,10))
  ax = plt.subplot(2,1,1)
  plt.title("Synaptic weight = " + str(weight) )
  for n in range(layer_number):
      ax.plot(t1, V1[n,0,:], color= "red", alpha = 0.25, linewidth = 0.5)
  ax.plot(t1, V1[0,0,:], color= "darkred", alpha = 1, linewidth = 1.5)
  plt.xlim(0,0.061)  
  ax.set_ylabel("Potential")
  
  ax = plt.subplot(2,1,2) #Raster
  plt.title("Spiking times for neurons")
  plt.ylim(0,layer_size*layer_number)
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Neuron Index")
  
  spikes = []
  for i in range(len(peaks)):
    for j in range(len(peaks[i])):
      plt.plot(t1[i],peaks[i][j], "|", color = "darkred")

  plt.tight_layout()
