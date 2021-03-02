#!/usr/bin/env bash
#!/bin/bash
"""
Created on Fri Mar 13 15:50:45 2020

@author: JDDL8382
"""
# coding: utf-8
import socket 
from threading import Thread
import sys
import torch.nn as nn
import time
import pickle
from syft.frameworks.torch.fl import utils
from numpy.random import seed
import numpy as np 
#import subprocess
import psutil
import os
import torch
#Get the process that running right now
torch.manual_seed(1)

pid = os.getpid()
#use psutil to detect this process

p = psutil.Process(pid)
#Return a float representing the current system-wide CPU utilization as a percentage
#First time you call the value is zero (as a baseline), the second it will compare with the value 
#called and give a result  
p.cpu_percent(interval=None, percpu=False)

#Funtion to convert seconds to hours
def secs2hours(secs):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    return "%d:%02d:%02d" % (hh, mm, ss)

#Creating architecture of the Neural Network model
class LSTM(nn.Module):
    def __init__(self, input_size=15, n_hidden=50, n_layers=1, output_size=5):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size=n_hidden, num_layers=n_layers)
        self.hidden = self.init_hidden()
        self.linear1 = nn.Linear(n_hidden, output_size)
        self.dropout1 = nn.Dropout(p=0.5)
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.n_layers,1,self.n_hidden),
                            torch.zeros(self.n_layers,1,self.n_hidden))
    def forward(self, x): 
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(x.view(x.size(0),1, -1), self.hidden)
        lstm_out = self.dropout1(lstm_out)
        predictions = self.linear1(lstm_out.view(len(lstm_out), -1))
        return predictions
    
#Arguments for the models
class Arguments:
    def __init__(self):
        #if there is no parameters passed the default is 5 
        self.communication_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        #if there is no parameters passed the default is 6 
        self.n_clients = int(sys.argv[2])  if len(sys.argv) > 2 else 6
        #if there is no parameters passed the default is 500 
        self.layers = 1
        self.units = 50
        self.n_steps_out = 5
        self.n_steps_in = 5
        self.n_features = 3


#Funtion to send the last model for all the clients that are connected to the server
def last_model(c,addr):
        
        start_time = time.time()
        with open('initial_model.sav', "rb") as r:
            #Sending the final model to all the clients
            print("sending the final model to ",addr, '\n')
            data = r.read()
            # check data length in bytes and send it to client
            data_length = len(data)
            c.send(data_length.to_bytes(4, 'big'))
            c.send(data)
        b = open("temps_communicate.txt", "a+")
        b.write('Time to send the last model to the client'+ str(addr) + ':' + str(secs2hours(time.time() - start_time))+ '\n')
        b.write('Time to send the last model to the client'+ str(addr) + ':' + str(time.time() - start_time)+ '\n')
        b.close()
             
        start_waiting = time.time()
        msg = int.from_bytes(c.recv(4), 'big')
        b = open("temps_communicate.txt", "a+")
        #Get the size of the bytes received from each client
        b.write("Bytes Received - Final Iteraction: " + str(msg) + '\n' )
        b.close()
        
        #receiving the final model form the clients
        print("receiving final model updated of the client", addr, '\n')
        while msg:
            # until there are bytes left...
            # fetch remaining bytes or 4096 (whatever smaller)
            rbuf = c.recv(min(msg, 4096))
            msg -= len(rbuf)
            # write to file
            
        #get the time to communicate with each client
        b = open("temps_communicate.txt", "a+")
        b.write("Time to receive final model from the client: " + str(addr) + ':' + str(secs2hours(time.time() - start_waiting)) + '\n' )
        b.write("Time to receive final model from the client: " + str(addr) + ':' + str(time.time() - start_waiting) + '\n' )
        b.close()
        
        #get the CPU and memory usage in the last communication round
        j = open("memory_cpu.txt", "a+")
        j.write('Final Iteration')
        j.write('physical memory use: (in MB)'+ str(p.memory_info()[0]/2.**20))
        j.write('percentage utilization of this process in the system' + str(p.cpu_percent(interval=None, percpu=False)))
        j.close()
            
def clientHandler(c, addr):
    #A thread for each client is created
    global count
    global iteration
    global clients
    try:
        msg = 0
        start_time = time.time()
        #Sending the models to the clients that were randomly choosen (ex = 4)
        with open('initial_model.sav', "rb") as r:
            print("opened initial model and sending to client",addr, '\n')
            data = r.read()
            # check data length in bytes and send it to client
            data_length = len(data)
            c.send(data_length.to_bytes(4, 'big'))
            c.send(data)
            
            
        print('time to send the federated model to the client',addr,':' ,time.time() - start_time, '\n')
        
        #Wait to receive the models from the client
        msg = int.from_bytes(c.recv(4), 'big')

        #Removing any previous models received from other communication rounds
        if os.path.exists('model'+str(count)+'.sav'):
            os.remove('model'+str(count)+'.sav')
     
        #Saving the model received from the client
        f = open('model'+str(count)+'.sav','wb')
        print("receiving model updated of the client", addr, '\n')
        count = count + 1
        #Only waiting 35 minutes to receive the model (hard constraint)
        c.settimeout(1800.0)
        while msg:
            # until there are bytes left...
            # fetch remaining bytes or 4096 (whatever smaller)
            rbuf = c.recv(min(msg, 4096))
            msg -= len(rbuf)
            # write to file
            print(msg)
            f.write(rbuf)
        f.close()
    
            
    except:
        #If there is an timeout or an inactive client that doesnt respond the connection 
        #is closed and the client is removed from the clients list
        print("Error. Data not sent to all clients.")
        c.close()
        clients.remove((c,addr))
        sys.exit()



args = Arguments()


n_timesteps, n_features, n_outputs = args.n_steps_in, args.n_features, args.n_steps_out
n_input = n_timesteps * n_features
#Initialize the model
initial_model = LSTM(n_input, args.units, args.layers, n_outputs)
#Defining the ealy stopping method

#Saving the  model in a file
filename = 'initial_model.sav'
pickle.dump(initial_model, open(filename, 'wb'))

#Defining the ip and port of the server that the clients will connect with
host = "0.0.0.0"
port = 80
  
#Making a socket to open communication
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
  
# binding the socket to the port and ip host 
s.bind((host, port))
# put the socket into listening mode (15 sec) 
s.listen(15) 
print("Ready for clients connection")

trds = []
#Initialize the clients url
# -*- coding: utf-8 -*-
clients = []
iteration = 0
count = 0
# seed random number generator
seed(1)

#Record all the clients connected and their ip adress in a file 
r = open("clients.txt", "a+")
for i in range(args.n_clients):
    #Do a socket communication beetween the clients 
    c, addr = s.accept() 
    print(addr, "is Connected")
    #Append the client to a clients list when its succesfully connected
    clients.append((c,addr))
    r.write('Client'+ str(i)+'\n')
    r.write('Adresse' + str(addr)+'\n')
r.close()

#Start execution until achieve maximum communication rounds
start_time = time.time()
while iteration < args.communication_rounds - 1:
    models = {}
    count = 0
    #Generate randomly numbers of integers that means the number of clients that will be conected
    #For example - 6 clients connected the training will happen for 4 clients
    #Our fraction was C=0.7
    values = np.random.choice(len(clients), int(0.7*len(clients)), replace=False)
    print("Clients " + str(values), '\n')
    print("Iteration Number " + str(iteration), '\n')
    
    iteration = iteration + 1
    for i in values:
        #Make a thread for each client to send them the model and wait for their response
        t = Thread(target=clientHandler, args = (clients[i][0], clients[i][1]))
        trds.append(t)
        t.start()
    
    #Only continue when all the threads are finalized 
    #- when all clients have responded (syncronous approach)
    for tr in trds:
        #join is to see if a thread is terminated
       tr.join()
       
    #Record the information of memory and cpu usage of the server
    r = open("memory_cpu.txt", "a+")
    r.write("Iteration Number " + str(iteration) + '\n')
    r.write('physical memory use: (in MB)'+ str(p.memory_info()[0]/2.**20)+ '\n')
    r.write('physical memory use: (in MB)'+ str(p.memory_percent())+ '\n')
    r.write('percentage utilization of this process in the system' + str(p.cpu_percent(interval=None))+ '\n')
    r.close()
    #Loading all the updated models received from the clients -we have for beacause we trained with 4 clients- 
    
    for i in range(int(0.7*len(clients))):
        print(i)
        models[i] = pickle.load(open('model'+str(i)+'.sav', 'rb'))
    

    #doing the federated avg
    federated_model = utils.federated_avg(models)
    
    #Saving the global model to be sent again to the clients
                    
    filename = 'initial_model.sav'
    pickle.dump(federated_model, open(filename, 'wb'))
    
#When all communication rounds end, the final model is sent to all the clients that are still connected
for client in clients:
    t = Thread(target=last_model, args = (client[0], client[1]))
    trds.append(t)
    t.start()


for tr in trds:
    tr.join()

#record the execution time of the execution

a = open("temps_execution.txt", "a+")
time_execution =  time.time() - start_time 
a.write("Time to execute: " + str(secs2hours(time_execution))+" of the client"+ '\n' )
a.close()



#To finalize the socket and the connections with the clients are closed

s.close()
for client in clients:
    client[0].close()

