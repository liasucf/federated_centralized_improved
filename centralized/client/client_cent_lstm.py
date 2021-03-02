# # Implementing LSTM Prediction of a Time Series in PyTorch
import os
import time
import socket
import pickle
import psutil
import pandas as pd
#import syft as sy
import numpy as np
import sys
import itertools
from sklearn.model_selection import TimeSeriesSplit
from numpy import array

def secs2hours(secs):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    return "%d:%02d:%02d" % (hh, mm, ss)

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

#Arguments for the models
class Arguments:
    def __init__(self):
        self.data = sys.argv[1]
        self.communication_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        self.seed = 1
        self.n_steps_out = 5
        self.n_steps_in = 5

args = Arguments()
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
## Loading data incrementaly 

host = "54.94.82.121"
port = 80
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.connect((host,port))
iteration = 0
list_data_time = []
data_class = []

data = pd.read_csv('Data/'+args.data +'.csv', sep=',')
data = data.iloc[0:5250]    

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

values = np.column_stack((data['Time'], data['temperature'], data['humidity'], data['tvoc'],data['co2'],data['class'] ))

data_time = values[:,0]
list_data_time.append(pd.to_datetime(data_time, format='%Y-%m-%d %H:%M:%S'))
list_data_time = array(list_data_time)

#taking out the time column
values = values [:,1:]
n_steps_in, n_steps_out = args.n_steps_in, args.n_steps_out

bull , y_class = split_sequences(values, n_steps_in , n_steps_out )

data_class.append(y_class)
data_class = array(data_class)

values = values[:, :-1]
#Creating the sliding window matrix

# convert into input/output
X,y = split_sequences(values, n_steps_in, n_steps_out)

# split into input and outputs
#X, y = serie.iloc[:, :-n_steps_out], serie.iloc[:, -n_steps_out:len(serie)]
n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]

n_input = n_timesteps * n_features

X = X.reshape((X.shape[0], n_input))

tscv = TimeSeriesSplit(n_splits=args.communication_rounds)
generator = tscv.split(X)

while True:
    
    msg = s.recv(1024)
    if (msg.decode("ascii") == 'OK'):
        print(msg.decode("ascii"))
        print(iteration)
        if os.path.exists('data_pickled.sav'):
            os.remove('data_pickled.sav')
                

        result = next(itertools.islice(generator, iteration + 1))
        print(result)
        train_index, test_index = result[0], result[1]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        data_train = np.column_stack((X_train, y_train))
        data_test = np.column_stack((X_test, y_test))

        time_data = list_data_time[:,test_index]
        class_data = data_class[:,test_index]
        data = []

        data.append(data_train)
        data.append(data_test)
        data.append(time_data)
        data.append(class_data)

        # ### Creating the sliding window matrix
        filehandler = open(b"data_pickled.sav","wb")
        pickle.dump(data,filehandler)
        filehandler.close()
    
        z = open("memory_cpu.txt", "a+")
        z.write("Iteration: " + str(iteration) + '\n')

    #Sending the updated model to the server    
        with open('data_pickled.sav', "rb") as r:
            start_time = time.time()        
            print("sending the data",  '\n')
            data_updated = r.read()
            # check data length in bytes and send it to client
            data_length = len(data_updated)

            s.send(data_length.to_bytes(4, 'big'))
            s.send(data_updated)
        r.close()
        z.write("Time to send data to server: " + str(time.time() -start_time ) + '\n')
        z.write("Time to send data to server: " + str(secs2hours(time.time() -start_time)) + '\n')
        z.write("Lenght of data in this iteration : " + str(len(data)) + '\n')
        #Collecting the information of memory and CPU usage
        z.write('percentage of memory use: '+ str(p.memory_percent())+ '\n')
        z.write('physical memory use: (in MB)'+ str(p.memory_info()[0]/2.**20))
        z.write('percentage utilization of this process in the system '+ str(p.cpu_percent(interval=None))+ '\n')
        z.write('percentage CPU '+ str(psutil.cpu_percent(interval=None, percpu=True))+ '\n')
        z.write('percentage CPU '+ str(psutil.cpu_percent(interval=None, percpu=False))+ '\n')
        z.close()
        
        iteration = iteration + 1

    if (msg.decode("ascii") == 'BYE'):
        sys.exit()
    
    
