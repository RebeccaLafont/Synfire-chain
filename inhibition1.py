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
in_number=30            #number of inhibitory neurons (1/5 of Ex neurons)
tau_neuron  = 10*ms    
tau_synapse = 5*ms   
tau_inhibition= 1*ms #lower than tau_neuron so frequency of spike higher
V_rest  = -60
V_reset = -55
V_spike = -50
layer_number = 10
layer_size = 15
neuron_nr1=layer_number*layer_size

#adaptation parameters
a=-0.5#*nS
b=0.5#*nA
tau_k=100*ms
Wk=np.zeros((150,n_iterations)) #adaptive potential

#inhibition potential
Vi=np.zeros((in_number,n_iterations))-60 #potential matrix for inhibatory neurons
#II=np.zeros((in_number, n_iterations)) #input to test V, V1, peaks = adapted_potential(weight)
Wie=np.zeros((30,150)) #weight matrix between I and E

#excitatory
V1 = np.zeros((layer_number, layer_size,n_iterations))-60 
V = V1.reshape(layer_number*layer_size,n_iterations) #150*n_iterations
I = np.zeros((layer_number*layer_size, n_iterations))
I[:15,:120] =30 #external current given only for the first layer, for 12ms
W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))
Wei=np.zeros((150,30))#weight matrix between E and I (has to be >95 for inhibitory population to be fully activated)


'''Create synaptic weight between each neuron of each excitatory layer, with the weight between
the neurons of the 4th and 5th layer being a variable parameter'''
def synaptic_weight1(layer_number,layer_size,weight):
  W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))
  for i in range(layer_number-1):
    W1[i+1,:,i,:]=1.5
  neuron_nr1=layer_number*layer_size #number of neuron per layer
  W1=W1.reshape(neuron_nr1,neuron_nr1)#shape of layers
  W1[60:75,45:60]=weight
  return W1, neuron_nr1




'''Create synaptic weight that connects subpopulation of inhibitory population to excitatory layer.
Each inhibitory neuron is connected to five excitatory neuron of the same layer'''
def weightIE(in_number,weightI):
    for i in range(in_number):
            Wie[i,i*5:(i+1)*5]=weightI
    return Wie

'''Create synaptic weight that connects excitaotry population to inhibitory subpopulation.
Five excitatory neuron is connected to one inhibitory neuron of the same layer. 
This acts as an input for the inhibiotyr subpopulation.'''
def weightEI(in_number,weightE):
    for j in range(in_number):
            Wei[j*5:(j+1)*5,j]=weightE
    return Wei


    
''' Create a function recording if there is a spike and the corresponding time
for the first timestep for the excitatory population'''
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

''' Create a function recording if there is a spike and the corresponding time
for the first timestep for the inhibitory population'''
def create_all_spikesI(Vi, dt, V_spike):
    spikes_ynI=(Vi[:,1]>=V_spike)
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
def postsyn_potential1(neuron_nr1, n_iterations, tau_synapse, all_spikes1, k, dt):
  E = np.zeros(neuron_nr1) #now neuron_nr is 150
  for m in range(neuron_nr1):
    e = 0

    f_times =np.unique(all_spikes1[m]) 
    
    for n in range(len(f_times)):
      et = np.exp(-((k+2)*dt-f_times[n])/tau_synapse) *np.sign(f_times[n])
      e=e+et 
    
    E[m] = e
  return E

'''Postsynaptic potential between neuron of excitatory layer and neuron of inhibiotry subpopulation'''
def postsyn_potentialEI(nr_neuron1, n_iterations, tau_synapse, all_spikes1, k, dt):
  E = np.zeros(nr_neuron1) 
  Eei = np.zeros(in_number) 
  for m in range(nr_neuron1):
    e = 0

    f_times =np.unique(all_spikes1[m]) 
    
    for n in range(len(f_times)):
      et = np.exp(-((k+2)*dt-f_times[n])/tau_synapse) *np.sign(f_times[n])
      e=e+et 
    
    E[m] = e
  for i in range(in_number): #make it shorter (1 on 5)
      Eei[i]=E[i*5]
  return Eei

'''Postsynaptic potential between neuron of inhibitory subpopulation and neurons of excitatory layer'''
def postsyn_potentialIE(in_number, n_iterations, tau_synapse, all_spikesI, k, dt):
  E = np.zeros(in_number) 
  Eie=np.zeros(neuron_nr1)
  for m in range(in_number):
    e = 0

    f_times =np.unique(all_spikesI[m]) 
    
    for n in range(len(f_times)):
      et = np.exp(-((k+2)*dt-f_times[n])/tau_synapse) *np.sign(f_times[n])
      e=e+et 
    
    E[m] = e
  for i in range(in_number):
      Eie[i*5:(i+1)*5]=E[i]
  return E



'''Final potential matrix for both population. Take in account spikes
of each population to activated the other.'''
def adapted_potential(weight,weightI,weightE):
  all_spikes1, spikes_yn1, last_spikes1=create_all_spikes1()
  peaks=[]
  
  spikes_ynI, last_spikesI, all_spikesI=create_all_spikesI(Vi, dt, V_spike)
  peaksI=[]
  #weight matrix
  W1,neuron_nr1=synaptic_weight1(layer_number,layer_size,weight)
  Wie=weightIE(in_number,weightI)
  Wei=weightEI(in_number,weightE)
  
  for k in range (n_iterations-100): #at each timestep
      #Eei=postsyn_potentialEI(neuron_nr1, n_iterations, tau_synapse, all_spikes1, k, dt)
      #Eie=postsyn_potentialIE(in_number, n_iterations, tau_synapse, all_spikesI, k, dt)
      E=postsyn_potential1(neuron_nr1,n_iterations,tau_synapse,all_spikes1,k,dt) 
      I_syn=np.dot(W1,E) #vector 1*150?
      #Iei=np.dot(Wei,Eei)
      #Iie=np.dot(Wie,Eie)
      c=0 
      d=0
      
      
      #I equation
      for j in range(in_number):
          if spikes_yn1[j*5]==1:#if E spikes activate I
              derivV1=(-(Vi[j,k+1]-V_rest) + Wei[j*5,j])/tau_inhibition  
              Vi[j,k+2]=Vi[j,k+1]+dt*derivV1
          else:
              derivV1=(-(Vi[j,k+1]-V_rest))/tau_inhibition 
              Vi[j,k+2]=Vi[j,k+1]+dt*derivV1
          
      #E equation    
      for j in range(in_number-1): 
          #to have a timestep of 5
          c=j*5
          d=(j+1)*5
          if spikes_ynI[j]==1: #if I spikes E inhibited
              deriv=(-(V[c:d,k+1]-V_rest)+I_syn[c:d]+I[c:d,k+1]-Wk[c:d,k+1]-Wie[j,j*5])/tau_neuron #Iie[j]
              V[c:d,k+2]=V[c:d,k+1]+deriv*dt
              Wk[c:d,k+2]=Wk[c:d,k+1]+(a*(V[c:d,k+1]-V_rest)-Wk[c:d,k+1])*dt/tau_k
          else:
              deriv=(-(V[c:d,k+1]-V_rest)+I_syn[c:d]+I[c:d,k+1]-Wk[c:d,k+1])/tau_neuron
              V[c:d,k+2]=V[c:d,k+1]+ deriv*dt
              Wk[c:d,k+2]=Wk[c:d,k+1]+(a*(V[c:d,k+1]-V_rest)-Wk[c:d,k+1])*dt/tau_k
     
      #reset if passes threshold E pop
      for i in range(len(all_spikes1)):
          if spikes_yn1[i]==1:
            V[i,k+2]=V_reset
            Wk[i,k+2]=Wk[i,k+2]+b #adaptation if E spikes
         
      #reset if passes threshold I pop
      for i in range(len(all_spikesI)):
        if spikes_ynI[i]==1:
            Vi[i,k+2]=V_rest
    
      #spikes for both population at this timestep
      spikes_yn1=(V[:,k+2]>=V_spike)
      spikes_ynI = (Vi[:,k+2]>=V_spike)
      
      
      #record spikes E pop
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
      
      #record spikes I pop
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
  return V, V1, peaks, Vi, peaksI



  

    
  '''Function plots the potential evolution and raster plot for both excitatory
  and inhibitory population (respectfully in red and blue)'''  
#PLOT
def PLOT_all(weight,weightI,weightE):
    V, V1, peaks, Vi,peaksI= adapted_potential(weight,weightI,weightE)
    fig = plt.figure(figsize = (8,10))
    ax = plt.subplot(4,1,1)
    plt.title("Evolution of potential in excitatory population with w="+str(weight)+" --- wIE="+str(weightI)+" --- wEI="+str(weightE) )
    for n in range(neuron_nr1):
        ax.plot(t, V[n,:], color= "red", alpha = 0.10, linewidth = 0.25)
    ax.plot(t, V[0,:], color= "darkred", alpha = 1, linewidth = 1.5)
    ax.set_ylabel("Potential (mV)")
    plt.xlim(0,0.06)
    
    ax = plt.subplot(4,1,2) #Raster
    plt.title("Spiking times for excitatory population")
    plt.ylim(0,neuron_nr1)
    plt.xlim(0,0.06)
    ax.set_ylabel("Neuron Index")
    spikes = []
    for i in range(len(peaks)):
      for j in range(len(peaks[i])):
        plt.plot(t[i],peaks[i][j]
                 , "|", color = "darkred")
        
    ax=plt.subplot(4,1,3)
    plt.title("Evolution of potential in inhibitory population with wIE="+str(weightI)+" --- wEI="+str(weightE))
    for n in range(in_number):
        ax.plot(t,Vi[n,:], color= "blue", alpha= 0.2, linewidth=0.25)
    ax.plot(t, Vi[0,:], color= "darkblue", alpha = 1, linewidth = 1.5)
    plt.xlim(0,0.06)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Potential (mV)")
    
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

'''Plots only excitatory population'''
def PLOT_E(weight,weightI,weightE):
    V, V1, peaks, Vi, peaksI = adapted_potential(weight,weightI,weightE)
    fig = plt.figure(figsize = (8,10))
    ax=plt.subplot(2,1,1)
    plt.title("Evolution of potential in Excitatory population  - w="+str(weight)+"-- wIE="+str(weightI)+"-- wEI="+str(weightE) )
    for n in range(neuron_nr1):
        ax.plot(t, V[n,:], color= "red", alpha = 0.25, linewidth = 0.25)
    ax.plot(t, V[0,:], color= "darkred", alpha = 1, linewidth = 1.5)
    ax.set_ylabel("Potential (mV)")
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