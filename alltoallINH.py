

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:37:16 2022

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
tau_inhibition= 5*ms #affects frequency of spikes
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
Vi=np.zeros((in_number,n_iterations))-60 #potential for inhibatory neurons
II=np.zeros((in_number, n_iterations)) 
II[:,:]=30 #constant input for all inhibitory population 
Wie=np.zeros((30,150)) #weight matrix between I and E

#excitatory
V1 = np.zeros((layer_number, layer_size,n_iterations))-60 
V = V1.reshape(layer_number*layer_size,n_iterations) 
I = np.zeros((layer_number*layer_size, n_iterations))
I[:15,:120] =30 #external current given only for the first layer, for 12ms
W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))


'''Create synaptic weight between each neuron of each excitatory layer, with the weight between
the neurons of the 4th and 5th layer being a variable parameter'''
def synaptic_weight1(layer_number,layer_size,weight):
  W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))
  for i in range(layer_number-1):
    W1[i+1,:,i,:]=1.5
  neuron_nr1=layer_number*layer_size 
  W1=W1.reshape(neuron_nr1,neuron_nr1)
  W1[60:75,45:60]=weight #change weight between 4th and 5th layer (goal: affect on syllable duration)
  return W1, neuron_nr1

'''Create synaptic weight that connects subpopulation of inhibitory population to excitatory layer.
Each inhibitory neuron is connected to five excitatory neuron of the same layer'''

def weightIE(in_number,weightI):
    for i in range(in_number):
            Wie[i,i*5:(i+1)*5]=weightI
    return Wie

''' Create a function recording if there is a spike and the corresponding time
for the first timestep for the excitatory population'''
def create_all_spikes1(V=V, dt=dt, threshold=V_spike):
  spikes_yn1=(V[:,1]>=threshold) #are there spikes at the first timestep?
  last_spikes1=np.zeros(len(spikes_yn1)) 
  for j in range (len(spikes_yn1)):
      if spikes_yn1[j]==1:
        last_spikes1[j]=dt#if there is, at what time?
      else:
        last_spikes1[j]=last_spikes1[j]

  all_spikes1=np.column_stack((last_spikes1,last_spikes1)) # (150*2)
  all_spikes1=all_spikes1.tolist()
  return all_spikes1, spikes_yn1, last_spikes1

''' Create a function recording if there is a spike and the corresponding time
for the first timestep for the inhibitory population'''
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

'''Postsynaptic potential between neuron of each excitatory layer'''
def postsyn_potential1(neuron_nr1, n, taus, all_spikes1, k, dt):
  E = np.zeros(neuron_nr1) 
  for m in range(neuron_nr1):
    e = 0
    f_times =np.unique(all_spikes1[m]) #keeps everytime their is a spike without repeating the times
    for n in range(len(f_times)):
      et = np.exp(-((k+2)*dt-f_times[n])/taus) *np.sign(f_times[n])#postsynaptic potential function
      e=e+et #add for each spikes
    E[m] = e
  return E

'''Final potential matrix for inhibitory population'''
def potentialI(weight):
    peaksI=[]
    spikes_ynI, last_spikesI, all_spikesI=create_all_spikesI(Vi, dt, V_spike)
    for k in range(n_iterations-100):
        derivV1=(-(Vi[:,k+1]-V_rest)+II[:,k+1])/tau_inhibition #potential equation
        Vi[:,k+2]=Vi[:,k+1]+dt*derivV1
        for i in range(len(all_spikesI)):
          if spikes_ynI[i]==1:
              Vi[i][k+2]=V_rest

        spikes_ynI = (Vi[:,k+2]>=V_spike) #record spikes for that timestep

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


'''Final potential matrix exictatory population, when inhibitory population spikes
excitatory population inhibited'''
def adapted_potential(weight,VI,weightI):
  all_spikes1, spikes_yn1, last_spikes1=create_all_spikes1()
  spikes_ynI, last_spikesI, all_spikesI=create_all_spikesI(Vi, dt, V_spike)
  VI,peaksI=potentialI(weight)
  peaks=[]
  for k in range (n_iterations-100): #for one k looking at matrix
      W1,neuron_nr1=synaptic_weight1(layer_number,layer_size,weight)
      E=postsyn_potential1(neuron_nr1,n_iterations,tau_synapse,all_spikes1,k,dt) 
      I_syn=np.dot(W1,E) #postsynaptic input
      Wie=weightIE(in_number,weightI)

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
          Wk[i,k+2]=Wk[i,k+2]+b #adpatation if their is a spike
        elif spikes_yn1[i] == 0:
          V[i,k+2] = V[i,k+2]

      #record spikes for both population at that timestep
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






#PLOT
'''Function plots the potential evolution and raster plot for both excitatory
and inhibitory population (respectfully in red and blue)'''
def PLOT_all(weight,weightI):
    Vi, peaksI=potentialI(weight)
    V, V1, peaks = adapted_potential(weight,Vi,weightI)
    fig = plt.figure(figsize = (8,10))
    ax = plt.subplot(4,1,1)
    plt.title("Evolution of potential in excitatory population with w="+str(weight)+" --- wIE="+str(weightI))
    for n in range(neuron_nr1):
        ax.plot(t, V[n,:], color= "red", alpha = 0.1, linewidth = 0.25)
    ax.plot(t, V[0,:], color= "darkred", alpha = 1, linewidth = 1.5)
    ax.set_ylabel("Potential (mV)")
    plt.xlim(0,0.06)

    ax = plt.subplot(4,1,2) #Raster
    plt.title("Spiking times for excitatory neurons")
    plt.ylim(0,neuron_nr1)
    plt.xlim(0,0.06)
    ax.set_ylabel("Neuron Index")
    spikes = []
    for i in range(len(peaks)):
      for j in range(len(peaks[i])):
        plt.plot(t[i],peaks[i][j]
                 , "|", color = "darkred")

    ax=plt.subplot(4,1,3)
    plt.title("Evolution of potential in inhibitory population wI="+str(weightI))
    for n in range(in_number):
        ax.plot(t,Vi[n,:], color= "blue", alpha= 0.2, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Potential (mV)")
    plt.xlim(0,0.06)
    
    ax = plt.subplot(4,1,4) #Raster
    plt.title("Spiking times for inhibitory population")
    plt.ylim(0,in_number)
    plt.xlim(0,0.06)
    ax.set_ylabel("Neuron Index")
    spikes = []
    for i in range(len(peaksI)):
      for j in range(len(peaksI[i])):
        plt.plot(t[i],peaksI[i][j]
                 , "|", color = "darkblue")

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