#!/bin/bash
#Set the base image
FROM arm64v8/python:3.6-slim

ARG data
ENV DATA_ENV=${data}
ARG communication_rounds
ENV ROUNDS_ENV=${communication_rounds}

#install requirements for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
         gcc g++ build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev gfortran liblapack-dev libopenblas-dev  pkg-config libffi-dev  libssl-dev 

RUN apt-get -y dist-upgrade

# Library components
RUN apt-get install -y \
    libavformat-dev libavcodec-dev libavdevice-dev libstdc++6 \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    libgle3  libopus-dev libvpx-dev libsrtp2-dev

WORKDIR /app
COPY ../requirements.txt requirements.txt
RUN pip3 install cython hypothesis pyyaml six
RUN pip3 install -r requirements.txt


#copy python script to the container folder app
COPY client_cent_lr.py client_cent_lr.py 
ADD ../Data Data/
RUN chmod +x client_cent_lr.py 


CMD python3 client_cent_lr.py $DATA_ENV $ROUNDS_ENV

