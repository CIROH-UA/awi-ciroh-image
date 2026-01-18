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
    apt-get install --yes -qq gnupg2 software-properties-common > /dev/null && \
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
RUN wget -q "https://sourceforge.net/projects/turbovnc/files/${TURBOVNC_VERSION}/turbovnc_${TURBOVNC_VERSION}_amd64.deb/download" -O turbovnc.deb \
 && apt-get update -qq --yes > /dev/null \
 && apt-get install -y ./turbovnc.deb > /dev/null \
 && apt-get remove -y light-locker > /dev/null \
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
RUN pip install spatialpandas easydev colormap colorcet duckdb dask_geopandas hydrotools sidecar && \
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
# SYMFLUENCE INSTALLATION SECTION - SPECIFIC VERSIONS
# ============================================================================

# Install basic build dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    zlib1g-dev \
    libbz2-dev \
    libexpat1-dev \
    libsqlite3-dev \
    libproj-dev \
    libtiff-dev \
    libpng-dev \
    libjpeg-dev \
    libgeos-dev \
    libopenblas-dev \
    liblapack-dev \
    bison \
    flex \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Install GCC 12 and gfortran 12 from Ubuntu toolchain PPA
# ============================================================================
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-12 g++-12 gfortran-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
        --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-12 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-12 && \
    rm -rf /var/lib/apt/lists/*

# ============================================================================
# Install OpenMPI 4.1.6 from source
# ============================================================================
WORKDIR /tmp
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz && \
    tar -xzf openmpi-4.1.6.tar.gz && \
    cd openmpi-4.1.6 && \
    ./configure \
        --prefix=/usr/local/openmpi-4.1.6 \
        CC=gcc-12 \
        CXX=g++-12 \
        FC=gfortran-12 && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf openmpi-4.1.6*

ENV PATH="/usr/local/openmpi-4.1.6/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/openmpi-4.1.6/lib:${LD_LIBRARY_PATH}"

# ============================================================================
# Install HDF5 1.14.x with parallel support
# ============================================================================
WORKDIR /tmp
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.3/src/hdf5-1.14.3.tar.gz && \
    tar -xzf hdf5-1.14.3.tar.gz && \
    cd hdf5-1.14.3 && \
    ./configure \
        --prefix=/usr/local/hdf5-1.14.3 \
        --enable-fortran \
        --enable-parallel \
        CC=/usr/local/openmpi-4.1.6/bin/mpicc \
        FC=/usr/local/openmpi-4.1.6/bin/mpif90 && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf hdf5-1.14.3*

ENV PATH="/usr/local/hdf5-1.14.3/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/hdf5-1.14.3/lib:${LD_LIBRARY_PATH}"

# ============================================================================
# Install NetCDF-C 4.9.x
# ============================================================================
WORKDIR /tmp
RUN wget https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.9.2.tar.gz && \
    tar -xzf v4.9.2.tar.gz && \
    cd netcdf-c-4.9.2 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=/usr/local/netcdf-4.9.2 \
        -DCMAKE_C_COMPILER=gcc-12 \
        -DENABLE_NETCDF_4=ON \
        -DENABLE_DAP=ON \
        -DHDF5_DIR=/usr/local/hdf5-1.14.3 \
        -DHDF5_C_LIBRARY=/usr/local/hdf5-1.14.3/lib/libhdf5.so \
        -DHDF5_HL_LIBRARY=/usr/local/hdf5-1.14.3/lib/libhdf5_hl.so \
        -DHDF5_INCLUDE_DIR=/usr/local/hdf5-1.14.3/include && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf netcdf-c-4.9.2 v4.9.2.tar.gz

ENV PATH="/usr/local/netcdf-4.9.2/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/netcdf-4.9.2/lib:${LD_LIBRARY_PATH}"
ENV CPATH="/usr/local/netcdf-4.9.2/include:${CPATH}"

# ============================================================================
# Install NetCDF-Fortran 4.6.x
# ============================================================================
WORKDIR /tmp
RUN wget https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v4.6.1.tar.gz && \
    tar -xzf v4.6.1.tar.gz && \
    cd netcdf-fortran-4.6.1 && \
    mkdir build && cd build && \
    export NCDIR=/usr/local/netcdf-4.9.2 && \
    export NFDIR=/usr/local/netcdf-fortran-4.6.1 && \
    export CPPFLAGS=-I${NCDIR}/include && \
    export LDFLAGS=-L${NCDIR}/lib && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=${NFDIR} \
        -DCMAKE_C_COMPILER=gcc-12 \
        -DCMAKE_Fortran_COMPILER=gfortran-12 \
        -DNETCDF_C_LIBRARY=${NCDIR}/lib/libnetcdf.so \
        -DNETCDF_INCLUDE_DIR=${NCDIR}/include && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf netcdf-fortran-4.6.1 v4.6.1.tar.gz

ENV PATH="/usr/local/netcdf-fortran-4.6.1/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/netcdf-fortran-4.6.1/lib:${LD_LIBRARY_PATH}"

# ============================================================================
# Install GDAL 3.9.4 with Python bindings
# ============================================================================
WORKDIR /tmp
RUN wget https://github.com/OSGeo/gdal/releases/download/v3.9.4/gdal-3.9.4.tar.gz && \
    tar -xzf gdal-3.9.4.tar.gz && \
    cd gdal-3.9.4 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=/usr/local/gdal-3.9.4 \
        -DCMAKE_C_COMPILER=gcc-12 \
        -DCMAKE_CXX_COMPILER=g++-12 \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DPython_EXECUTABLE=$(which python3) && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf gdal-3.9.4*

ENV PATH="/usr/local/gdal-3.9.4/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/gdal-3.9.4/lib:${LD_LIBRARY_PATH}"
ENV GDAL_DATA="/usr/local/gdal-3.9.4/share/gdal"

# Install GDAL Python bindings via pip
RUN pip install --no-cache-dir GDAL==3.9.4 --no-binary GDAL \
    --global-option=build_ext \
    --global-option="-I/usr/local/gdal-3.9.4/include" \
    --global-option="-L/usr/local/gdal-3.9.4/lib" || \
    pip install --no-cache-dir GDAL==3.9.4

# ============================================================================
# Install R 4.5.x
# ============================================================================
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc && \
    add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" && \
    apt-get update && \
    apt-get install -y r-base r-base-dev && \
    rm -rf /var/lib/apt/lists/*

# If R 4.5 is not available, build from source
WORKDIR /tmp
RUN R_VERSION=$(R --version | head -1 | grep -oP '\d+\.\d+\.\d+') && \
    if [ "$(echo "$R_VERSION < 4.4" | bc -l)" -eq 1 ]; then \
        wget https://cran.r-project.org/src/base/R-4/R-4.5.0.tar.gz && \
        tar -xzf R-4.5.0.tar.gz && \
        cd R-4.5.0 && \
        ./configure \
            --prefix=/usr/local \
            --enable-R-shlib \
            --with-blas \
            --with-lapack && \
        make -j$(nproc) && \
        make install && \
        cd /tmp && rm -rf R-4.5.0*; \
    fi

# ============================================================================
# Install CDO 2.2.x
# ============================================================================
WORKDIR /tmp
RUN wget https://code.mpimet.mpg.de/attachments/download/29449/cdo-2.2.2.tar.gz && \
    tar -xzf cdo-2.2.2.tar.gz && \
    cd cdo-2.2.2 && \
    ./configure \
        --prefix=/usr/local/cdo-2.2.2 \
        --with-netcdf=/usr/local/netcdf-4.9.2 \
        --with-hdf5=/usr/local/hdf5-1.14.3 \
        CC=gcc-12 \
        CXX=g++-12 \
        CPPFLAGS="-I/usr/local/netcdf-4.9.2/include -I/usr/local/hdf5-1.14.3/include" \
        LDFLAGS="-L/usr/local/netcdf-4.9.2/lib -L/usr/local/hdf5-1.14.3/lib" && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf cdo-2.2.2*

ENV PATH="/usr/local/cdo-2.2.2/bin:${PATH}"

# ============================================================================
# Set compiler environment variables for SYMFLUENCE
# ============================================================================
ENV CC=gcc-12
ENV CXX=g++-12
ENV FC=gfortran-12
ENV MPICC=/usr/local/openmpi-4.1.6/bin/mpicc
ENV MPICXX=/usr/local/openmpi-4.1.6/bin/mpicxx
ENV MPIFC=/usr/local/openmpi-4.1.6/bin/mpif90

# ============================================================================
# Install SYMFLUENCE
# ============================================================================
RUN mkdir -p /opt/symfluence
WORKDIR /opt/symfluence

# Clone SYMFLUENCE repository
# Replace with actual repository URL
RUN git clone https://github.com/your-org/symfluence.git . || \
    echo "WARNING: SYMFLUENCE repository URL needs to be updated"

# Create Python virtual environment
RUN python3 -m venv /opt/symfluence/.venv

# Install SYMFLUENCE dependencies
RUN if [ -f requirements.txt ]; then \
        /opt/symfluence/.venv/bin/pip install --upgrade pip && \
        /opt/symfluence/.venv/bin/pip install -r requirements.txt; \
    fi

# Run installer
RUN if [ -f ./symfluence ]; then \
        chmod +x ./symfluence && \
        ./symfluence --install || true; \
    fi

ENV PATH="/opt/symfluence/.venv/bin:/opt/symfluence:${PATH}"

# ============================================================================
# Cleanup and finalize
# ============================================================================
WORKDIR /tmp
RUN ldconfig

# Update custom Jupyter Lab settings
RUN sed -i 's/\"default\": true/\"default\": false/g' /srv/conda/envs/notebook/share/jupyter/labextensions/@axlair/jupyterlab_vim/schemas/@axlair/jupyterlab_vim/plugin.json 2>/dev/null || true

USER ${NB_USER}
WORKDIR /home/${NB_USER}
