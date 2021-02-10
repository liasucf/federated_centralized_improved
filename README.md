# federated_centralized
A work to compare the Federated and the Centralized Machine Learning approaches.

This work is composed of two docker containers that represent a client and a server. 
To build this project start by building the containers with the docker files given.
Run first the docker of the server and when it is prepared to accept clients run them.


## Here are some main arguments that the docker file accepts as parameters. 

- data_name: The name of the data file that has the data used by the client.
              In our example the data names are: 9953, 9958, 9959, 9994, 12487, 12843.
              This parameter is required to run the docker container.
              
 - number_communication_rounds: this consits in the number of time that the server will perform the model aggregation in the Federated Learning. It can be seen as a sort of iteration. This parameters is needed in the server and client docker.
 By default this number is 5.
 
- n_clients: this is the number of clients (instances) that will participate in the training. If you have 5 virtual machines connected to the server you will have 5 clients. 
By default this number is 6

- n_epochs: This is the number of epochs for our deep learning model to be trained. By default this number is 500. 
  
## To run the docker container of the server

docker build . -t server_fed      
docker run -e ROUNDS_ENV=<number_communication_rounds> -e CLIENTS_ENV=<n_clients> -p 80:80 -v /home/ec2-user/federated/server:/app -it server_fed
## To run the docker container of the client 

docker build . -t client_fed        
docker run -e DATA_ENV=<data_name> -e ROUNDS_ENV=<number of communication rounds> -e EPOCHS_ENV=<n_epochs> --mount source=myvol2,target=/app -it client_fed       
  
  
## To test in local network you need to create a network in Docker 
Ex:           
docker network create --subnet=172.17.0.0/16 my-network     
Server:   docker run --net my-network --ip 172.18.0.2 -e ROUNDS_ENV=<number_communication_rounds> -e CLIENTS_ENV=<n_clients> -p 8000:8000 -it client_fed        
Client:   docker run --net my-network -e DATA_ENV=<data_name> -e ROUNDS_ENV=<number_communication_rounds> -e EPOCHS_ENV=<n_epochs> -it client_fed 
