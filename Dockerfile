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

# Install desktop environment packages
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
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y nodejs

# Install TurboVNC
ARG TURBOVNC_VERSION=2.2.6
RUN curl -fL "https://sourceforge.net/projects/turbovnc/files/${TURBOVNC_VERSION}/turbovnc_${TURBOVNC_VERSION}_amd64.deb/download" -o turbovnc.deb \
 && apt-get update -qq --yes \
 && ls -lh turbovnc.deb \
 && file turbovnc.deb \
 && apt-get install -y ./turbovnc.deb \
 && apt-get remove -y light-locker \
 && rm ./turbovnc.deb \
 && ln -s /opt/TurboVNC/bin/* /usr/local/bin/ \
 && rm -rf /var/lib/apt/lists/*

RUN mamba install -n ${CONDA_ENV} -y websockify

# Install jupyter-remote-desktop-proxy
RUN export PATH=${NB_PYTHON_PREFIX}/bin:${PATH} \
 && npm install -g npm@7.24.0 \
 && pip install --no-cache-dir \
        https://github.com/jupyterhub/jupyter-remote-desktop-proxy/archive/main.zip

# Install jupyterlab extensions
RUN pip install jupyterlab_vim jupyter-tree-download

# Install Google Cloud SDK
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

# Install various Python packages
RUN pip install --no-cache-dir dask==2025.12.0 distributed==2025.12.0 spatialpandas easydev colormap colorcet duckdb dask_geopandas hydrotools sidecar && \
    pip install --upgrade colorama && \
    pip install nb_black==1.0.5

# HydroShare packages
RUN pip install hsclient[all]==1.1.6 pydantic==2.7.* && \
    pip install -U --no-cache-dir --upgrade-strategy only-if-needed \
    git+https://github.com/hydroshare/nbfetch.git@v0.6.4 && \
    jupyter server extension enable --py nbfetch --sys-prefix

# Install additional packages
RUN pip install google-cloud-bigquery dataretrieval hsfiles-jupyter && \
    python -m hsfiles_jupyter

# ============================================================================
# SYMFLUENCE INSTALLATION SECTION
# ============================================================================

# Install SYMFLUENCE system dependencies
# libproj-dev and libgeos-dev are required per SYMFLUENCE docs
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    gfortran \
    cmake \
    libopenmpi-dev \
    openmpi-bin \
    libgdal-dev \
    gdal-bin \
    libhdf5-dev \
    libhdf5-openmpi-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libproj-dev \
    libgeos-dev \
    cdo \
    r-base \
    r-base-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set compiler and GDAL include path environment variables
ENV CC=gcc
ENV CXX=g++
ENV FC=gfortran
ENV MPICC=mpicc
ENV MPICXX=mpicxx
ENV MPIFC=mpif90
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install SYMFLUENCE into the shared conda environment so it is available to all users
RUN pip install symfluence

# Install external model binaries (SUMMA, mizuRoute, FUSE, NGEN, TauDEM, etc.)
RUN symfluence binary install

# Register SYMFLUENCE as a Jupyter kernel available to all JupyterHub users
RUN python -m ipykernel install --name "symfluence" --display-name "Python (SYMFLUENCE)" --sys-prefix

# ============================================================================
# END SYMFLUENCE INSTALLATION
# ============================================================================

# Update custom Jupyter Lab settings
RUN sed -i 's/\"default\": true/\"default\": false/g' /srv/conda/envs/notebook/share/jupyter/labextensions/@axlair/jupyterlab_vim/schemas/@axlair/jupyterlab_vim/plugin.json

# Switch back to notebook user
USER ${NB_USER}

# Set working directory back to home
WORKDIR /home/${NB_USER}
