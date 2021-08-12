#!/bin/bash
#Set the base image
FROM arm64v8/python:3.6-slim

ARG data
ARG communication_rounds
ARG n_epochs

ENV DATA_ENV=${data}
ENV ROUNDS_ENV=${communication_rounds}
ENV EPOCHS_ENV=${n_epochs}

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

RUN pip install cython hypothesis pyyaml six
RUN pip install  Pillow==6.2.2
RUN pip install numpy==1.18.5

ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/openblas/libblas.so

#creates work dir   
WORKDIR /app

RUN  python3 -m pip install https://github.com/liasucf/Auxiliar/raw/main/torch-1.4.0-cp36-cp36m-linux_aarch64.whl
RUN  python3 -m pip install https://github.com/liasucf/Auxiliar/raw/main/torchvision-0.5.0-cp36-cp36m-linux_aarch64.whl
RUN python3 -m pip install av

COPY ../requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip3 install tblib syft_proto websocket websockets phe
RUN pip3 install --no-dependencies syft==0.2.6

#copy python script to the container folder app
COPY client_fed_svr.py client_fed_svr.py
ADD ../Data Data/
RUN chmod +x client_fed_svr.py
RUN echo $(ls)
RUN echo $(ls Data)

CMD python client_fed_svr.py $DATA_ENV $ROUNDS_ENV $EPOCHS_ENV
