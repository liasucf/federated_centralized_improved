/# federated_centralized
A work to compare the Federated and the Centralized Machine Learning approaches.

This work is composed of two docker images that represent a client and a server, in our tests we have used 
AWS AMV8 Instances to build the docker images. 
Run first the docker of the server and when it is prepared to accept clients run them.


## Here are some main arguments that the docker file accepts as parameters. 

- data_name: The name of the data file that has the data used by the client.
              In our example the data names are: 9953, 9958, 9959, 9994, 12487, 12483.
              This parameter is required to run the docker container.
              
 - number_communication_rounds: this consits in the number of time that the server will perform the model aggregation in the Federated Learning. It can be seen as a sort of iteration. This parameters is needed in the server and client docker.
 By default this number is 5.
 
- n_clients: this is the number of clients (instances) that will participate in the training. If you have 5 virtual machines connected to the server you will have 5 clients. 
By default this number is 6

- n_epochs: This is the number of epochs for our deep learning model to be trained. By default this number is 500. 
  
## To run the docker container of the server

docker build . -t server_fed      
docker run -e ROUNDS_ENV=<number_communication_rounds> -e CLIENTS_ENV=<n_clients> -p 80:80 -v /home/ec2-user/federated_centralized/federated/server:/app -it server_fed

## To run the docker container of the client 

docker build . -t client_fed        
docker run -e DATA_ENV=<data_name> -e ROUNDS_ENV=<number of communication rounds> -e EPOCHS_ENV=<n_epochs> -v /home/ec2-user/federated_centralized/federated/client:/app -it client_fed       
  
  
