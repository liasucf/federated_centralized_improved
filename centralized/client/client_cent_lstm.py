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

def secs2hours(secs):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    return "%d:%02d:%02d" % (hh, mm, ss)
#Arguments for the models
class Arguments:
    def __init__(self):
        self.data = sys.argv[1]
        self.communication_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        self.seed = 1


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
while True:
    
    msg = s.recv(1024)
    if (msg.decode("ascii") == 'OK'):
        print(msg.decode("ascii"))
        print(iteration)
        if os.path.exists('data_pickled.sav'):
            os.remove('data_pickled.sav')
                
        data = pd.read_csv('Data/'+args.data +'.csv', sep=',')
        data = data.iloc[0:5300]    
        
        size_data = int(5300/args.communication_rounds)
        
        data = data.iloc[(0 + size_data*iteration):(size_data*(iteration+1))]  
        
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
        # ### Creating the sliding window matrix
        print(data)
        filehandler = open(b"data_pickled.sav","wb")
        pickle.dump(values,filehandler)
        filehandler.close()
        
        z = open("memory_cpu.txt", "a+")
        z.write("Iteration: " + str(iteration) + '\n')

    #Sending the updated model to the server    
        with open('data_pickled.sav', "rb") as r:
            start_time = time.time()        
            print("sending the updated model",  '\n')
            data_updated = r.read()
            # check data length in bytes and send it to client
            data_length = len(data_updated)

            s.send(data_length.to_bytes(4, 'big'))
            s.send(data_updated)
        r.close()
        z.write("Time to send data to server: " + str(time.time() -start_time ) + '\n')
        z.write("Time to send data to server: " + str(secs2hours(time.time() -start_time)) + '\n')
        z.write("Lenght of data in this iteration : " + str(data) + '\n')
        #Collecting the information of memory and CPU usage
        z.write('percentage of memory use: '+ str(p.memory_percent())+ '\n')
        z.write('physical memory use: (in MB)'+ str(p.memory_info()[0]/2.**20))
        z.write('percentage utilization of this process in the system '+ str(p.cpu_percent(interval=None))+ '\n')
        z.close()
        
        iteration = iteration + 1

    if (msg.decode("ascii") == 'BYE'):
        sys.exit()
    
    
