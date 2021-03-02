# # Implementing LSTM Prediction of a Time Series in PyTorch
import os
import time
import socket
import pickle
import psutil
import torch
import torch.nn as nn
import pandas as pd
#import syft as sy
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import array
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from skorch.callbacks import EarlyStopping
from skorch import NeuralNetRegressor
from math import sqrt
import itertools
import matplotlib.dates as mdates

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

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    

def secs2hours(secs):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    return "%d:%02d:%02d" % (hh, mm, ss)
#Arguments for the models
class Arguments:
    def __init__(self):
        self.data = sys.argv[1]
        self.communication_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        self.epochs = int(float(sys.argv[3])) if len(sys.argv) > 3 else 200
        self.seed = 2
        self.lr = 0.01
        self.batch_size = 8
        self.patience = 100
        self.momentum =  0.09
        self.threshold = 0.0003
        self.n_steps_out = 5
        self.n_steps_in = 5
        

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


args = Arguments()
torch.manual_seed(args.seed)
#Get the process that running right now
pid = os.getpid()
#use psutil to detect this process
p = psutil.Process(pid)
#Return a float representing the current system-wide CPU utilization as a percentage
#First time you call the value is zero (as a baseline), the second it will compare with the value 
#called and give a result  
p.cpu_percent(interval=None, percpu=False)

# ## DataSet

# ### Charging the data
## Loading data incrementaly 
data = pd.read_csv('Data/'+args.data +'.csv', sep=',')


data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S')
data['temperature'] = data['temperature'].astype(str).astype(float)
data['humidity'] = data['humidity'].astype(str).astype(float)
data['tvoc'] = data['tvoc'].astype(str).astype(float)
data['co2'] = data['co2'].astype(str).astype(float)

data.sort_values(by=['Time'], inplace = True)
data.reset_index(drop= True, inplace = True)
#Creating the classes for classification
#data = data.iloc[0:args.n_samples]
bins = [50, 1000, 1500, 8000]
labels = ["Good","Minor Problemns","Hazardous"]
data['class'] = pd.cut(data['co2'].values, bins=bins, labels=labels)

values = np.column_stack((data['temperature'], data['humidity'], data['tvoc'],data['co2'], data['class']))
# ### Creating the sliding window matrix
n_steps_in, n_steps_out = args.n_steps_in, args.n_steps_out

bull , y_class = split_sequences(values, n_steps_in , n_steps_out )
values = values[:, :-1]


X, y = split_sequences(values, n_steps_in, n_steps_out)


n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]   
n_input = n_timesteps * n_features  
X = X.reshape((X.shape[0], n_input))

tscv = TimeSeriesSplit(n_splits=args.communication_rounds)
generator = tscv.split(X)
host = "54.94.82.121"
port = 80

#Making a socket for communication and connecting with the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.connect((host,port))
number = 0
current_round = 0
increment = 0
#A forever loop to always be listening to the server 
while True:
    msg = 0
  
 
    #Receiving the model from the server
    start_time = time.time()
    print('Connected to server', '\n')
    #Message received
    msg = int.from_bytes(s.recv(4), 'big')

    #Saving the message received with the models parameters 
    print("Saving the model received", '\n')
    f = open('model_rec.sav','wb')
    while msg:
        # until there are bytes left...
        # fetch remaining bytes or 4094 (whatever smaller)
        rbuf = s.recv(min(msg, 4096))
        msg -= len(rbuf)
        # write to file
        f.write(rbuf)
    f.close()
    
    g = open("time_communicate.txt", "a+")
    g.write("Iteration: " + str(number) + '\n' )
    g.write('Time to receive the model from the server' + str(time.time() - start_time) + '\n')
    g.write('Time to receive the model from the server' + str(secs2hours(time.time() - start_time)) + '\n')
    g.close()

    increment = increment + 1
    while increment > 0:

        current_round = current_round + 1
        print(current_round)
        result = next(itertools.islice(generator, current_round))
        train_index, test_index = result[0], result[1]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        fig_split, ax = plt.subplots(figsize=(8,5))
        ax.plot(data['Time'][train_index],y_train[:,n_steps_out-1] ,linestyle='-', linewidth=2, label='Train', color='#E65132')
        ax.plot(data['Time'][np.hstack((train_index,test_index))], [None for i in y_train[:,n_steps_out-1]] + [x for x in y_test[:,n_steps_out-1]] , linestyle='-', linewidth=2, label='Test', color='#6156FA')
        date_form = mdates.DateFormatter("%d %b")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.set_xlim(data['Time'][np.hstack((train_index,test_index))].min(), data['Time'][np.hstack((train_index,test_index))].max())
        ax.set_ylabel('eCO2 (ppm)')
        ax.set_xlabel('Time')
        ax.legend();
        fig_split.savefig('train_test_split_plot'+str(number)+'.png', bbox_inches='tight')
        
        # identify outliers in the training dataset
        lof = LocalOutlierFactor()
        yhat = lof.fit_predict(X_train)
        # select all rows that are not outliers
        mask = yhat != -1
        X_train, y_train = X_train[mask, :], y_train[mask]
        # summarize the shape of the updated training dataset

        y_class_test = y_class[test_index]
        
        
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.transform(X_test)
        
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)
        
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
        
        
        model = pickle.load(open('model_rec.sav', 'rb'))
        
        
        early = EarlyStopping(patience=args.patience, threshold= args.threshold )

        #Using the model with the NeuralNetRegressor to configure parameters
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
        
        start_training = time.time()
        net.fit(X_train, y_train)

        #saving the training time
        b = open("train_temps.txt", "a+")
        b.write("Iteration: " + str(number) + '\n' )
        b.write("Lenght X: " + str(len(X)) + '\n' )
        b.write("Lenght X train: " + str(len(X_train)) + '\n' )
        b.write("Lenght X test: " + str(len(X_test)) + '\n' )
        b.write(" Time to train: " + str(secs2hours(time.time() - start_training))  + '\n' )
        b.write( " Time to train: " + str(time.time() - start_training)  + '\n' )
        b.close()
        
        
        # visualize the loss as the network trained
        # plotting training and validation loss
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
        fig.savefig('loss_plot'+str(number)+'.png', bbox_inches='tight')

        y_pred = net.predict(X_test)

        a = open("test_losses.txt", "a+")
        a.write("Number: " + str(number) + '\n')
        a.write("MSE loss: " + str(mean_squared_error(y_test, y_pred)) + " MAE loss: " + str(mean_absolute_error(y_test, y_pred))  + '\n' )
        a.write("RMSE loss: " + str(sqrt(mean_squared_error(y_test, y_pred))) + " MAPE loss: " + str(mean_absolute_percentage_error(y_test.numpy(), y_pred))+ '\n' ) 
        a.close()

        target = scaler_y.inverse_transform(y_pred)
        real = scaler_y.inverse_transform(y_test)
 

        fig1, ax = plt.subplots(figsize=(10,5))
        ax.plot(data['Time'][test_index], real[:,n_steps_out-1],color='#6156FA', label='Test' )
        ax.plot(data['Time'][test_index], target[:,n_steps_out-1],color='#FFBB69', label = 'Prediction')
        
        date_form = mdates.DateFormatter("%d %b")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.set_xlabel('Time')
        ax.set_ylabel('eCO2 (ppm)')
        ax.set_title('Concentration of eCO2 over time')
        ax.legend()
        fig1.savefig('prediction_round_'+str(number)+'.png', bbox_inches='tight')


        y_pred =  scaler_y.inverse_transform(y_pred)


        bins = [50, 1000, 1500, 8000]
        labels = ["Good","Minor Problemns","Hazardous"]

        
        y_class_pred = pd.cut( y_pred.reshape(-1), bins=bins, labels=labels).astype(str)    
        y_class_pred = y_class_pred.reshape((y_pred.shape[0], n_outputs))


        print(np.unique(y_class_test))
        print(np.unique(y_class_pred))
        
        h = open("classification_accuracy.txt", "a+")
        h.write("Number: " + str(number) + '\n')
        h.write("Labels Real: " + str(np.unique(y_class_test)) + '\n')
        h.write("Labels Predicted: " + str(np.unique(y_class_pred)) + '\n')
        h.write("Accuracy of Classification on test set: " + str(accuracy_score(y_class_test.reshape(-1),y_class_pred.reshape(-1))) + '\n')
        h.write("Confusion Matrix: " + str(confusion_matrix(y_class_test.reshape(-1), y_class_pred.reshape(-1))) + '\n')
        cm = confusion_matrix(y_class_test.reshape(-1), y_class_pred.reshape(-1))
        h.write("True Positive: " + str(np.diag(cm)) + " Support for each label " + str(np.sum(cm, axis = 1)) + '\n')
        h.write("Recall: " + str(np.diag(cm) / np.sum(cm, axis = 1)) + " Precision: " + str(np.diag(cm) / np.sum(cm, axis = 0)) + '\n')
        h.write("Recall Mean: " + str(np.mean(np.diag(cm) / np.sum(cm, axis = 1))) + " Precision Mean: " + str(np.mean(np.diag(cm) / np.sum(cm, axis = 0))) + '\n')
        h.close()

        increment = increment - 1

    
    #Collecting the information of memory and CPU usage
    z = open("memory_cpu.txt", "a+")
    z.write("Number: " + str(number) + '\n')
    z.write('percentage of memory use: '+ str(p.memory_percent())+ '\n')
    z.write('physical memory use: (in MB)'+ str(p.memory_info()[0]/2.**20))
    z.write('percentage utilization of this process in the system '+ str(p.cpu_percent(interval=None, percpu=False))+ '\n')
    z.close()
    
    
    #Saving the model updated
    number = number + 1
    filename = 'model.sav'
    pickle.dump(model, open(filename, 'wb'))

    #Sending the updated model to the server    
    with open("model.sav", "rb") as r:
                
        print("sending the updated model",  '\n')
        data_updated = r.read()
        # check data length in bytes and send it to client
        data_length = len(data_updated)
        s.send(data_length.to_bytes(4, 'big'))
        s.send(data_updated)
        print('Time to send updated model to the server',time.time() - start_time , '\n')
    r.close()


