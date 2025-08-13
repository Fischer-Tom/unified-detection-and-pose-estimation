FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y vim python3 python3-pip ffmpeg libsm6 libxext6 build-essential git libprotobuf-dev protobuf-compiler libopencv-dev python3-opencv libgflags-dev libeigen3-dev wget bzip2 ca-certificates curl software-properties-common lsb-release gpg && apt-get clean && rm -rf /varlib/apt/lists/*


# Setup GLOG and GFLAGS for Progressive-X
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor -o /etc/apt/trusted.gpg.d/kitware.gpg

# Add Kitware's repository to the sources list
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"

# Update again and install the kitware-archive-keyring package
RUN apt-get update && \
    apt-get install -y kitware-archive-keyring && \
    rm /etc/apt/trusted.gpg.d/kitware.gpg

RUN apt update && apt install cmake -y

# Install GFLAGS and GLOG (Required for Progressive-X)
RUN git clone https://github.com/google/glog.git /tmp/glog && \
    cd /tmp/glog && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/glog


# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install progressive-x with submodules
RUN git clone --recurse-submodules https://github.com/Fischer-Tom/progressive-x.git /tmp/progressive-x && \
    cd /tmp/progressive-x && \
    pip install . && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/progressive-x


RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
RUN pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.7.0cu126

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/pyprogressivex:$LD_LIBRARY_PATH

ENV SHELL=/bin/bash
RUN ln -sf /bin/bash /bin/sh