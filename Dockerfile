# This Dockerfile aims to provide a Pangeo-style image with the VNC/Linux Desktop feature
# It was constructed by following the instructions and copying code snippets laid out
# and linked from here:
# https://github.com/2i2c-org/infrastructure/issues/1444#issuecomment-1187405324

FROM pangeo/pangeo-notebook:2024.04.08

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH ${NB_PYTHON_PREFIX}/bin:$PATH

# Needed for apt-key to work
RUN apt-get update -qq --yes > /dev/null && \
    apt-get install --yes -qq gnupg2 > /dev/null && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get -y update \
 && apt-get install -y dbus-x11 \
   firefox \
   xfce4 \
   xfce4-panel \
   xfce4-session \
   xfce4-settings \
   xorg \
   xubuntu-icon-theme \
   curl \
 && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
#RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - \
#    && apt-get install -y nodejs

# Install TurboVNC (https://github.com/TurboVNC/turbovnc)
#ARG TURBOVNC_VERSION=2.2.6
#RUN wget -q "https://sourceforge.net/projects/turbovnc/files/${TURBOVNC_VERSION}/turbovnc_${TURBOVNC_VERSION}_amd64.deb/download" -O turbovnc.deb \
# && apt-get update -qq --yes > /dev/null \
# && apt-get install -y ./turbovnc.deb > /dev/null \
 # remove light-locker to prevent screen lock
# && apt-get remove -y light-locker > /dev/null \
# && rm ./turbovnc.deb \
# && ln -s /opt/TurboVNC/bin/* /usr/local/bin/ \
# && rm -rf /var/lib/apt/lists/*

#RUN mamba install -n ${CONDA_ENV} -y websockify

# Install jupyter-remote-desktop-proxy with compatible npm version
#RUN export PATH=${NB_PYTHON_PREFIX}/bin:${PATH} \
# && npm install -g npm@7.24.0 \
# && pip install --no-cache-dir \
#        https://github.com/jupyterhub/jupyter-remote-desktop-proxy/archive/main.zip

# Install jupyterlab_vim extension
#RUN pip install jupyterlab_vim

# TO download the folder/files:
#RUN pip install jupyter-tree-download 

# Install Google Cloud SDK (gcloud, gsutil)
#RUN apt-get update && \
#    apt-get install -y curl gnupg && \
#    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
#    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
#    apt-get update -y && \
#    apt-get install google-cloud-sdk -y

# Gfortran support
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y software-properties-common \
    && apt-get install -yq python3.9 python3-pip python3-wheel make cmake \
    libnetcdf-dev liblapack-dev libopenblas-dev g++ git libssl-dev valgrind gdb \
    gfortran gcc-multilib libnetcdff-dev libcoarrays-dev libopenmpi-dev && \
    apt-get clean -q

# Install dataretrieval package
#RUN pip install dataretrieval
RUN pip install cmake

WORKDIR /code

# Install the C++ Actor Framework 0.18.6
RUN wget https://github.com/actor-framework/actor-framework/archive/refs/tags/0.18.6.tar.gz
RUN tar -xvf 0.18.6.tar.gz
WORKDIR /code/actor-framework-0.18.6
RUN ./configure
WORKDIR /code/actor-framework-0.18.6/build
RUN make -j 4
RUN make test
RUN make install

WORKDIR /code

# Install Sundials
RUN wget https://github.com/LLNL/sundials/releases/download/v7.0.0/sundials-7.0.0.tar.gz
RUN tar -xzf sundials-7.0.0.tar.gz
RUN mkdir sundials
WORKDIR /code/sundials
RUN mkdir /usr/local/sundials
RUN mkdir builddir
WORKDIR /code/sundials/builddir
RUN cmake ../../sundials-7.0.0 -DBUILD_FORTRAN_MODULE_INTERFACE=ON \
        -DCMAKE_C_COMPILER=/usr/bin/gcc \
        -DCMAKE_Fortran_COMPILER=gfortran \
        -DCMAKE_INSTALL_PREFIX=/usr/local/sundials \
        -DEXAMPLES_INSTALL_PATH=/code/sundials/instdir/examples
RUN make
RUN make install

# Change workdir for when we attach to this container
WORKDIR /Summa-Actors

USER ${NB_USER}
