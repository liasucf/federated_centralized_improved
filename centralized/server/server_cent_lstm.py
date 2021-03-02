#!/usr/bin/env python
# coding: utf-8

# In[104]:
import pickle
import socket 
import torch
import torch.nn as nn
import time
import sys
import numpy as np
from math import sqrt
from numpy.random import seed
import pandas as pd
from threading import Thread
#import syft as sy
import matplotlib.pyplot as plt
from numpy import array
import os
import psutil
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor

# # Implementing MLP Prediction of a Time Series in PyTorch

# ## Defining the Neural Network

# In[134]:
def secs2hours(secs):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    return "%d:%02d:%04d" % (hh, mm, ss)

#Creating architecture of the Neural Network model
class LSTM(nn.Module):
    def __init__(self, input_size=35, n_clients=2, n_hidden=50, n_layers=1, output_size=5):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_clients = n_clients
        self.lstm = nn.LSTM(input_size, hidden_size=n_hidden, num_layers=n_layers)
        self.hidden = self.init_hidden()
        self.linear1 = nn.Linear(n_hidden, output_size)
        self.dropout1 = nn.Dropout(p=0.5)
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.n_layers, self.n_clients ,self.n_hidden),
                            torch.zeros(self.n_layers, self.n_clients, self.n_hidden))
    def forward(self, x): 
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = self.dropout1(lstm_out)
        predictions = self.linear1(lstm_out)
        return predictions


# ## Defining functions that will be used (train, test, sliding window, model evaluation)

# In[135]:
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Arguments for the models
#Arguments for the models
class Arguments:
    def __init__(self):
        
        self.communication_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        #if there is no parameters passed the default is 6 
        self.n_clients = int(sys.argv[2])  if len(sys.argv) > 2 else 6
        #if there is no parameters passed the default is 500 
        self.epochs = int(float(sys.argv[3])) if len(sys.argv) > 3 else 200
        
        self.lr = 0.01
        self.batch_size = 8
        self.seed = 1
        self.patience = 100
        self.momentum =  0.09
        self.threshold = 0.0003
        self.layers = 1
        self.units = 50
        self.n_steps_out = 5
        self.n_steps_in = 5
        self.n_features = 3


def receive_data(c,addr):
    try: 
            
        message = "OK"
        c.send(message.encode('ascii'))
        
        
        start_time = time.time()
        print("receiving the data from client", addr, '\n')

        #Only waiting 35 minutes to receive the model (hard constraint)
        c.settimeout(1800.0)
 
        msg = int.from_bytes(c.recv(4), 'big')
        print("Number of bytes received of data:" + str(msg))
        #Saving the model received from the client
        f = open('data'+str(addr)+'.sav','wb')
        while msg:
            # until there are bytes left...
            # fetch remaining bytes or 4096 (whatever smaller)
            rbuf = c.recv(min(msg, 4096))
            msg -= len(rbuf)
            # write to file
            f.write(rbuf)
        f.close()
        b = open("time_communicate.txt", "a+")
        b.write("Iteration: " + str(iteration) + '\n')
        b.write("Time to receive data from: "+str(addr) + "is " + str(time.time() - start_time))
        b.write("Time to receive data from: "+str(addr) + "is " + str(secs2hours(time.time() - start_time)))
        b.close()

    except:
        #If there is an timeout or an inactive client that doesnt respond the connection 
        #is closed and the client is removed from the clients list
        print("Error. Data not received from all clients.")
        c.close()
        clients.remove((c,addr))
        sys.exit()

def end_connection(c,addr):
    message = "BYE"
    c.send(message.encode('ascii'))
    print('Ending connection')
    c.close()
    clients.remove((c,addr))

args = Arguments()
torch.manual_seed(args.seed)
#Get the process that running right now
pid = os.getpid()
#use psutil to detect this process
p = psutil.Process(pid)
#Return a float representing the current system-wide CPU utilization as a percentage
#First time you call the value is zero (as a baseline), the second it will compare with the value 
#called and give a result  
p.cpu_percent(interval=None)

# ## DataSet

# ### Charging the data
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
global iteration
iteration = 0
# seed random number generator
seed(1)
# In[59]:
list_of_addresses = []
#Record all the clients connected and their ip adress in a file 
r = open("clients.txt", "a+")
for i in range(args.n_clients):
    #Do a socket communication beetween the clients 
    c, addr = s.accept() 
    print(addr, "is Connected")
    #Append the client to a clients list when its succesfully connected
    clients.append((c,addr))
    list_of_addresses.append(addr)
    r.write('Client'+ str(i)+'\n')
    r.write('Adresse' + str(addr)+'\n')
r.close()
n_steps_out = args.n_steps_out
start_time = time.time()
while iteration < args.communication_rounds - 1:
    print(iteration)
    models = {}
    
    for client in clients:
        t = Thread(target=receive_data, args = (client[0], client[1]))
        trds.append(t)
        t.start()
        
    #Only continue when all the threads are finalized (all clients sent their models)
    #- when all clients have responded (syncronous approach)
    for tr in trds:
       tr.join()
       
    
    seq_X_train = []
    seq_y_train = []
    seq_X_test = []
    seq_y_test = []
    seq_time = []
    seq_class = []

    df_X_train = pd.DataFrame(seq_X_train) 
    df_y_train = pd.DataFrame(seq_y_train) 
    df_X_test = pd.DataFrame(seq_X_test) 
    df_y_test = pd.DataFrame(seq_y_test) 
    

    for a in list_of_addresses:
        file = open('data'+str(a)+'.sav','rb')
        data = pickle.load(file)
        #getting the time colum
        data_train = data[0]
        data_test = data[1]
        time_data = data[2]
        class_data = data[3]


        X_train, y_train = data_train[:,:(data_train.shape[1] - args.n_steps_out)] , data_train[:,(data_train.shape[1] - args.n_steps_out):]
        X_test, y_test = data_test[:,:(data_train.shape[1] - args.n_steps_out)] , data_test[:,(data_train.shape[1] - args.n_steps_out):]

        for i in range(X_train.shape[1]):        
            df_X_train[str(a)+'_'+str(i)] = X_train[:,i]
            df_X_test[str(a)+'_'+str(i)] = X_test[:,i]

        for i in range(y_train.shape[1]):
            df_y_train[str(a)+'_co2_'+str(i)] = y_train[:,i]
            df_y_test[str(a)+'_co2_'+str(i)] = y_test[:,i]

        df_X_train.append(df_X_train)
        df_y_train.append(df_y_train)
        df_X_test.append(df_X_test)
        df_y_test.append(df_y_test)
        seq_time.append(time_data)
        seq_class.append(class_data)

    df_time = np.array(seq_time)
    df_class = np.array(seq_class)
    df_time = df_time.reshape((df_time.shape[0], df_time.shape[2]))
    df_class = df_class.reshape((df_class.shape[0], df_class.shape[2] ,df_class.shape[3]))

    X_train = df_X_train.values
    y_train = df_y_train.values
    X_test = df_X_test.values
    y_test = df_y_test.values


    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X_train)
    yhat = lof.fit_predict(X_train)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, y_train = X_train[mask, :], y_train[mask]

    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    

    n_data = args.n_clients 
    n_input = args.n_steps_in * args.n_features
    n_outputs = args.n_steps_out

    X_train = X_train.reshape((X_train.shape[0], n_data,  n_input))
    X_test = X_test.reshape((X_test.shape[0], n_data, n_input ))
    
    y_train = y_train.reshape((y_train.shape[0],  n_data,n_outputs))
    y_test = y_test.reshape((y_test.shape[0], n_data, n_outputs))
    
    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    

    #Initialize the model
    model = LSTM(n_input,args.n_clients, args.units,  args.layers, n_outputs)

    early = EarlyStopping(patience=args.patience, threshold= args.threshold )
    
    net = NeuralNetRegressor(
    model,
    max_epochs=args.epochs,
    lr=args.lr,
    batch_size = args.batch_size,
    optimizer__momentum=args.momentum,
    iterator_train__shuffle=False,
    iterator_valid__shuffle=False
    #callbacks=[early]
    )
    
    start_training_time = time.time()
    net.fit(X_train, y_train)
    
    
    # In[72]:
    z = open("memory_cpu.txt", "a+")
    z.write("Iteration: " + str(iteration) + '\n')
    z.write("Lenght of X_train: " + str(len(X_train)) + '\n')
    z.write("Tempo para treinar" + str(secs2hours(time.time() - start_training_time)))
    z.write('percentage of memory use: '+ str(p.memory_percent())+ '\n')
    z.write('physical memory use: (in MB)'+ str(p.memory_info()[0]/2.**20))
    z.write('percentage utilization of this process in the system '+ str(p.cpu_percent(interval=None))+ '\n')
    z.write('percentage CPU '+ str(psutil.cpu_percent(interval=None, percpu=True))+ '\n')
    z.write('percentage CPU '+ str(psutil.cpu_percent(interval=None, percpu=False))+ '\n')
    z.close()
    
    
    # get training and validation loss
    epochs = [i for i in range(len(net.history))]
    train_loss = net.history[:,'train_loss']
    valid_loss = net.history[:,'valid_loss']
    fig = plt.figure(figsize=(8,5))
    plt.plot(epochs,train_loss,'g-');
    plt.plot(epochs,valid_loss,'r-');
    plt.title('Training Loss Curves');
    plt.xlabel('Epochs');
    plt.ylabel('Mean Squared Error');
    plt.legend(['Train','Validation']);
    fig.savefig('loss_plot'+str(iteration)+'.png', bbox_inches='tight')
    
    # In[73]:
    
    
    y_pred = net.predict(X_test)

    a = open("test_losses.txt", "a+")    
    for i in range(args.n_clients):
        a.write('MSE for Client '+str(i) + ' is: ' + str(mean_squared_error(y_test[:,i,:], y_pred[:,i,:]))+ '\n' )
        a.write('MAE for Client '+str(i) + ' is: ' + str(mean_absolute_error(y_test[:,i,:], y_pred[:,i,:]))+ '\n' )
        a.write("RMSE loss: " + str(sqrt(mean_squared_error(y_test[:,i,:], y_pred[:,i,:]))) + " MAPE loss: " + str(mean_absolute_percentage_error(y_test[:,i,:].numpy(), y_pred[:,i,:]))+ '\n' ) 
    a.close()
    
    target = scaler_y.inverse_transform(y_pred.reshape((y_pred.shape[0], n_data * n_outputs)))
    real = scaler_y.inverse_transform(y_test.reshape((y_test.shape[0], n_data * n_outputs)))

    target = target.reshape((target.shape[0], n_data ,  n_outputs))
    real = real.reshape((real.shape[0], n_data ,  n_outputs))

    for i in range(args.n_clients):
        fig1, ax = plt.subplots(figsize=(10,5))
        plt.plot(df_time[i,:], real[:,i,n_steps_out-1],color='#6156FA', label='Test' )
        plt.plot(df_time[i,:], target[:,i,n_steps_out-1],color='#FFBB69', label = 'Prediction')
        date_form = mdates.DateFormatter("%d %b")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.set_xlabel('Time')
        ax.set_ylabel('eCO2 (ppm)')
        ax.set_title('Concentration of eCO2 over time')
        ax.legend()
        fig1.savefig('prediction_client_'+str(i)+'.png', bbox_inches='tight')
        
    y_pred = y_pred.reshape((y_pred.shape[0], n_data * n_outputs))
    y_test_normal = scaler_y.inverse_transform(y_test.reshape((y_test.shape[0], n_data * n_outputs)))
    y_pred =  scaler_y.inverse_transform(y_pred)


    y_class_test = df_class
    #data_class = data_class.reshape((data_class.shape[1], args.n_clients ,  n_outputs))

    bins = [50, 1000, 1500, 8000]
    labels = ["Good","Minor Problemns","Hazardous"]
    
    
    y_class_pred = pd.cut( y_pred.reshape(-1), bins=bins, labels=labels).astype(str)    
    y_class_pred = y_class_pred.reshape((y_pred.shape[0], n_data ,  n_outputs))
    

    h = open("classification_accuracy.txt", "a+")
    h.write("Number: " + str(iteration) + '\n')
    for i in range(args.n_clients):

        h.write('Accuracy of the client :'+str(i+1) + ' is: ' + str(accuracy_score(y_class_test[i,:,:].reshape(-1), y_class_pred[:,i,:].reshape(-1)))+ '\n' )
        h.write("Labels Real: " + str(np.unique(y_class_test[i,:,:])) + '\n')
        h.write("Labels Predicted: " + str(np.unique(y_class_pred[:,i,:])) + '\n')
        h.write("Confusion Matrix: " + str(confusion_matrix(y_class_test[i,:,:].reshape(-1), y_class_pred[:,i,:].reshape(-1))) + '\n')
        cm = confusion_matrix(y_class_test[i,:,:].reshape(-1), y_class_pred[:,i,:].reshape(-1))
        h.write("True Positive: " + str(np.diag(cm)) + " Support for each label " + str(np.sum(cm, axis = 1)) + '\n')
        h.write("Recall: " + str(np.diag(cm) / np.sum(cm, axis = 1)) + " Precision: " + str(np.diag(cm) / np.sum(cm, axis = 0)) + '\n')
        h.write("Recall Mean: " + str(np.mean(np.diag(cm) / np.sum(cm, axis = 1))) + " Precision Mean: " + str(np.mean(np.diag(cm) / np.sum(cm, axis = 0))) + '\n')
    h.close()
    
    
        
    iteration = iteration + 1

print("Tempo de execução: " + str(secs2hours(time.time() - start_time)))

trs = []
for client in clients:
    t = Thread(target=end_connection, args = (client[0], client[1]))
    trs.append(t)
    t.start()

for tr in trs:
    tr.join()

for client in clients:
    client[0].close()

s.close()
