# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Forward Euler in the function of integrate-and-fire model
import numpy as np
import matplotlib.pyplot as plt
import random
#parameters
tau1=0.01 #for 1rst neuron
tau=0.1
dt=0.001 #ms
duration=1#s
I1=2 #input only for 1rst neuron
I=0 #mV
n=int(duration/dt) #nbre pts
weight1=1
weight2=2 #mV
def f(v,I,tau):
    return (I-v)/tau
def ForwardEuler(f,duration,dt,I1,tau1): #seul moment ou I diff 0, input seulement pr premier
    v=np.zeros(n+1)
    t=np.zeros(n+1)
    v[0]=0
    peaks=[]
    for i in range(n):
        v[i+1]=v[i]+dt*f(v[i],I1,tau1)
        t[i+1]=t[i]+dt
        if v[i]>=1:
            v[i+1]=0
            peaks.append(i)
        else:
            None
    return [t,v,peaks]
#neuron1
t=ForwardEuler(f,duration,dt,I1,tau1)[0]
v=ForwardEuler(f,duration,dt,I1,tau1)[1]
peaks=ForwardEuler(f,duration,dt,I1,tau1)[2]

'''Equation for second neuron, takes in a ccount the post synaptic potential'''
def ForwardEulertarget(weight1,f,duration,dt,I,tau,peaks):
    v1=np.zeros(n+1)
    t1=np.zeros(n+1)
    for i in range(n):
        v1[i+1]=v1[i]+dt*f(v1[i],I,tau)
        t1[i+1]=t1[i]+dt
        if i in peaks:
            v1[i+1]=v1[i+1]+weight1*0.2 #0.2= postsynaptic potential
        if v1[i]>=1:
            v1[i+1]=0
    return[t1,v1]

'''Plots the evolution of potential of the synapse and a raster plot of the spikes'''
def plot_SYN(weight1, weight2, duration, dt, I, tau1):
  v, t, peaks = ForwardEuler(f, duration , dt, I, tau=0.01) # Neuron 1
  v1, t1 = ForwardEulertarget(weight1, f, duration, dt, 0, tau1, peaks) # Neuron 2
  v2, t2 = ForwardEulertarget(weight2, f, duration, dt, 0, tau1, peaks) # Neuron 3
  fig, ax = plt.subplots(1, 1)  
  ax.set_xlabel("Time(s)")
  ax.set_ylabel("Voltage(V)")
  plt.plot(v, t, linestyle='dashed', alpha=0.5, color='black')
  plt.plot(v1, t1)
  plt.plot(v2, t2)
  plt.xlim(0, 0.1)
  plt.legend(["Neuron 1","Neuron 2", "Neuron 3"])



#Chain
#Parameters & Initialization
ms = 0.001
dt = 0.01*ms
duration  = 60*ms
tau = 10*ms #tau of the fire and integrate neuros
taus = 5*ms #tau of the synapse
n = int(duration/dt) #number of steps
neuron_nr = 11 #number of neurons
threshold = -50
rest = -60
np.random.seed(1) #initialization of the random
V0 = np.zeros((neuron_nr,n+1))-60
t = np.linspace(0, duration, n+1)
I0 = tau*(threshold-rest)/dt + 1 #current given only for the first layer




'''Creates a weight matrix that connects neurons of the chain to the previous and follwoing. 
The weight between the 4th and 5th layer is taken as a parameter.'''
def synaptic_weight(neuron_nr,weight):
  W=np.zeros((neuron_nr,neuron_nr))
  for i in range(neuron_nr):
    for k in range(neuron_nr):
      if k==i+1:
        W[k][i]=43
  W[4][3]=weight
  W[0][-1]=0
  return W

#only for the first timestep
V0[0,1] = I0/tau*dt + rest

''' Create a function recording if there is a spike and the corresponding time
for the first timestep for the  chain'''
def create_all_spikes(V0=V0, I0=I0, tau=tau, dt=dt, rest=rest, threshold=threshold):
    spikes_yn=(V0[:,1]>=threshold) #a boolean array that confirms spike or not

    last_spikes=spikes_yn*dt
    
    for j in range(len(spikes_yn)):
        if spikes_yn[j] == 1:#if there is a spike, find the time when it happened
            last_spikes[j]=dt 
        else:
            last_spikes[j]=0
          
  #create a list that takes in all the arrays of spike times
    all_spikes = np.column_stack((last_spikes,last_spikes))
    all_spikes = all_spikes.tolist() 
  
    return all_spikes, spikes_yn, last_spikes

all_spikes, spikes_yn, last_spikes = create_all_spikes()
 
'''Postsynaptic potential between neuron of the chain'''
def postsyn_potential(neuron_nr, n, taus, all_spikes, k, dt):  
  E = np.zeros(neuron_nr)
  for m in range(neuron_nr):
    e = 0    
    #from the list of all recorded spike times, take into account only the ones
    #registered once
    f_times =np.unique(all_spikes[m]) 
    for n in range(len(f_times)):
      if f_times[n]==0:
          et=0
      #find the postsynaptic potential for every previous spike 
      else:
          et = np.exp(-(k*dt - f_times[n])/taus)          
          #add postsynaptic potentials for every previous spike to find the postsynaptic 
          #potential of the neuron
          e=e+et #add potential from other's neurone's spikes     
    E[m] = e
  return E

'''Final potential matrix for the chain. Take in account weights and postsynaptic potential.'''
def weight_related_result(weight, neuron_nr=neuron_nr, n=n, taus=taus, dt=dt, threshold=threshold):
  #use the create_all_spikes function to create them for the first ts
  all_spikes, spikes_yn, last_spikes = create_all_spikes()
  peaks = []
  for k in range(n-1):
    #calculate postsynaptic potential
    E=postsyn_potential(neuron_nr, n, taus, all_spikes, k, dt)
    #build the weight matrix
    W = synaptic_weight(neuron_nr, weight)
    #complete based on the formula dv/dt, careful because there is a matrix multiplication, 
    #hint: use np.dot
    derivV=(1/tau)*((-(V0[:,k+1]-rest))+np.dot(W,E))
    #FE
    V0[:,k+2]=V0[:,k+1]+dt*derivV
    #set the next ts to 0 if there is a spike; hint : check at spikes_yn
    for i in range(len(all_spikes)):
      if spikes_yn[i]==1:
          V0[i][k+2]=rest
    #check if the threshold was reached       
    spikes_yn = (V0[:,k+2]>=threshold)
    #record spikes
    A = np.argwhere(spikes_yn == True)
    # last_spikes = iteration * dt
    for j in range(len(spikes_yn)):
      if spikes_yn[j] == 1:
        last_spikes[j] = (k+2)*dt
      else:
        last_spikes[j] = last_spikes[j]
                
    for m in range(neuron_nr):
      if last_spikes[m] not in all_spikes[m]:
        all_spikes[m].append(last_spikes[m])
    peaks.append(A)
  return V0,peaks

'''Function plots the potential evolution and raster plot for the chain'''  
def plot_CHAIN(weight):
      neuron_nr=11
      V0, peaks = weight_related_result(weight)
      fig = plt.figure(figsize = (8,10))
      ax = plt.subplot(2,1,1)
      plt.title("Synaptic weight = " + str(weight) + "mV")
      for i in range(neuron_nr):
        ax.plot(t,V0[i,:]
                , color= "red", alpha = 0.25, linewidth = 0.5)
    #only the second neuron's voltage in time 
      ax.plot(t,V0[3,:]
              , color= "darkred", alpha = 1, linewidth = 1.5)
      plt.xlim(0, duration)  
      ax.set_ylabel("Potential")
      ax = plt.subplot(2,1,2) #raster
      plt.title("Spiking times for neurons")
      plt.xlim(0, duration)  
      plt.ylim(0, neuron_nr)
      ax.set_xlabel("Time (s)")
      ax.set_ylabel("Neuron Index")
      spikes = []
      for i in range(len(peaks)):
        for j in range(len(peaks[i])):
          plt.plot(t[i],peaks[i][j]
                   , "|", color = "darkred")
    
      plt.tight_layout()


#SYNFIRE CHAIN
#constants
s = second = 1.0
ms = millisecond = 0.001
debug = False

# simulation parameters
seed = 1
dt = 0.1*ms
duration1  = 300 * ms
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
I[:15,:50] = 30  #external current given only for the first layer, for 5ms
W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))
noise_mean = 0
noise_std = 2
noise_scale = np.sqrt(10*ms)

noise = np.random.normal(noise_mean, noise_std, n_iterations)
#plt.plot(noise*noise_scale)
#plt.plot(noise)

'''Create synaptic weight between each neuron of each excitatory layer, with the weight between
the neurons of the 4th and 5th layer being a variable parameter'''
def synaptic_weight1(layer_number,layer_size,weight):
  W1 = np.zeros((layer_number, layer_size, layer_number, layer_size))
  for i in range(layer_number-1):
    W1[i+1,:,i,:]=3
  neuron_nr1=layer_number*layer_size #number of neuron per layer
  W1=W1.reshape(neuron_nr1,neuron_nr1)#shape of layers
  W1[60:75,45:60]=weight
  return W1, neuron_nr1

W1,neuron_nr1=synaptic_weight1(layer_number,layer_size,1.5)

''' Create a function recording if there is a spike and the corresponding time
for the first timestep'''
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

'''Postsynaptic potential between neuron of each layer'''
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

'''Final potential matrix for take in account synaptic weights and post synaptic potential.'''
def weight_related_result1(weight):
  all_spikes1, spikes_yn1, last_spikes1=create_all_spikes1()
  peaks=[]
  for k in range (1000): #for one k looking at matrix
      W1,neuron_nr1=synaptic_weight1(layer_number,layer_size,weight) #matrix 150*150
      E=postsyn_potential1(neuron_nr1,n_iterations,tau_synapse,all_spikes1,k,dt) 
      I_syn=np.dot(W1,E) #vector 1*150?
      noise=np.random.normal(noise_mean,noise_std,neuron_nr1)
      #dv/dt
      derivV1=(-(V[:,k+1]-V_rest)+I_syn+I[:,k+1]+noise_scale*noise)/tau_neuron
      #FE
      V[:,k+2]=V[:,k+1]+dt*derivV1
      #reset if passes threshold
      for i in range(len(all_spikes1)):
        if spikes_yn1[i]==1:
          V[i,k+2]=V_reset
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

'''Function plots the potential evolution and raster for each layer of the synfire chain'''  
def plot_SYNFIRE(weight):
  V, V1, peaks = weight_related_result1(weight)
  fig = plt.figure(figsize = (8,10))
  ax = plt.subplot(2,1,1)
  plt.title("Synaptic weight = " + str(weight) +  "   ---   Noise std =" +str(noise_std) )
  for n in range(layer_number):
      ax.plot(t1, V1[n,0,:], color= "red", alpha = 0.25, linewidth = 0.5)
  ax.plot(t1, V1[0,0,:], color= "darkred", alpha = 1, linewidth = 1.5)
  plt.xlim(0,0.061)  
  ax.set_ylabel("Potential")
  
  

  
  ax = plt.subplot(2,1,2) #Raster
  plt.title("Spiking times for neurons")
  plt.ylim(0,layer_size*layer_number)
  plt.xlim(0,0.061)
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Neuron Index")
  
  spikes = []
  for i in range(len(peaks)):
    for j in range(len(peaks[i])):
      plt.plot(t1[i],peaks[i][j], "|", color = "darkred")

  plt.tight_layout()