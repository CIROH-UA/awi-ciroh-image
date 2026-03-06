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
 && apt-get install -y ./turbovnc.deb \
 && apt-get remove -y light-locker \
 && rm ./turbovnc.deb \
 && ln -s /opt/TurboVNC/bin/* /usr/local/bin/ \
 && rm -rf /var/lib/apt/lists/*

RUN mamba install -n ${CONDA_ENV} -y websockify
# Keep compiled numeric stack conda-managed to avoid ABI conflicts (pyarrow vs numpy)
# Stay on NumPy 1.x for compatibility with older compiled wheels.
RUN mamba install -n ${CONDA_ENV} -y -c conda-forge \
    "numpy<2" pandas pyarrow

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
RUN pip install --no-cache-dir dask==2025.12.0 distributed==2025.12.0 spatialpandas easydev colorcet duckdb dask_geopandas hydrotools sidecar && \
    pip install --no-cache-dir --no-deps colormap && \
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

# TODO: The following line is a debugging statement that can be removed.
RUN python -c "import numpy, pandas; import pyarrow; print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('pyarrow', pyarrow.__version__)"
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

# Pin SYMFLUENCE data/binary install to a fixed, world-readable path.
# Without this, binary_service._resolve_default_data_dir() falls back to
# a sibling of CWD at build time (unpredictable), and users at runtime
# would have a different CWD so the binaries would never be found.
ENV SYMFLUENCE_DATA_DIR=/opt/symfluence/data
ENV SYMFLUENCE_CODE_DIR=/opt/symfluence

RUN mkdir -p /opt/symfluence/data && chmod -R 755 /opt/symfluence

# Create a dedicated conda env for SYMFLUENCE to avoid conflicts with the base env
RUN mamba create -n symfluence -y python=3.11 ipykernel && \
    /srv/conda/envs/symfluence/bin/pip install \
        git+https://github.com/DarriEy/SYMFLUENCE.git@main

# Register the SYMFLUENCE kernel into the notebook env's prefix so ALL
# JupyterHub users see it — kernel spec points back to the symfluence env's Python
RUN /srv/conda/envs/symfluence/bin/python -m ipykernel install \
    --prefix=/srv/conda/envs/notebook \
    --name "symfluence" \
    --display-name "Python (SYMFLUENCE)"

# Install external model binaries into $SYMFLUENCE_DATA_DIR (/opt/symfluence/data).
# The ENV vars above are set image-wide, so all JupyterHub users resolve
# the same path at runtime without needing to set anything themselves.
RUN /srv/conda/envs/symfluence/bin/symfluence binary install

# ============================================================================
# END SYMFLUENCE INSTALLATION
# ============================================================================

# Update custom Jupyter Lab settings
RUN sed -i 's/\"default\": true/\"default\": false/g' /srv/conda/envs/notebook/share/jupyter/labextensions/@axlair/jupyterlab_vim/schemas/@axlair/jupyterlab_vim/plugin.json

# Switch back to notebook user
USER ${NB_USER}

# Set working directory back to home
WORKDIR /home/${NB_USER}
