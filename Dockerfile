# Headless AWI CIROH image with SYMFLUENCE.
# This variant keeps the current notebook + SYMFLUENCE workflow but removes
# desktop/VNC components and Google Cloud SDK to reduce image size.

FROM pangeo/pangeo-notebook:2024.04.08

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=${NB_PYTHON_PREFIX}/bin:$PATH

# Keep compiled numeric stack conda-managed to avoid ABI conflicts.
RUN mamba install -n ${CONDA_ENV} -y -c conda-forge \
    websockify \
    "numpy<2" \
    pandas \
    pyarrow

# Install notebook-focused Python packages.
RUN pip install --no-cache-dir \
    dask==2025.12.0 \
    distributed==2025.12.0 \
    spatialpandas \
    easydev \
    colorcet \
    duckdb \
    dask_geopandas \
    hydrotools \
    sidecar \
    jupyterlab_vim \
    jupyter-tree-download \
    nb_black==1.0.5 \
    google-cloud-bigquery \
    dataretrieval \
    hsfiles-jupyter && \
    pip install --no-cache-dir --no-deps colormap && \
    pip install --no-cache-dir hsclient[all]==1.1.6 pydantic==2.7.* && \
    pip install -U --no-cache-dir --upgrade-strategy only-if-needed \
    git+https://github.com/hydroshare/nbfetch.git@v0.6.4 && \
    pip install --no-cache-dir --upgrade colorama

RUN jupyter server extension enable --py nbfetch --sys-prefix && \
    python -m hsfiles_jupyter

# ============================================================================
# SYMFLUENCE INSTALLATION SECTION
# ============================================================================

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
    libudunits2-dev \
    libexpat1-dev \
    cdo \
    r-base \
    r-base-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc
ENV CXX=g++
ENV FC=gfortran
ENV MPICC=mpicc
ENV MPICXX=mpicxx
ENV MPIFC=mpif90
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

ENV SYMFLUENCE_DATA_DIR=/opt/symfluence/data
ENV SYMFLUENCE_CODE_DIR=/opt/symfluence
ENV SYMFLUENCE_ENV=/srv/conda/envs/symfluence

RUN mkdir -p /opt/symfluence/data && chmod -R 755 /opt/symfluence

RUN mamba create -n symfluence -y -c conda-forge \
        python=3.11 \
        ipykernel \
        "boost-cpp>=1.79" && \
    ${SYMFLUENCE_ENV}/bin/pip install \
        git+https://github.com/DarriEy/SYMFLUENCE.git@main && \
    ${SYMFLUENCE_ENV}/bin/pip uninstall -y \
        triton \
        nvidia-cublas-cu12 \
        nvidia-cuda-cupti-cu12 \
        nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 \
        nvidia-cudnn-cu12 \
        nvidia-cufft-cu12 \
        nvidia-cufile-cu12 \
        nvidia-curand-cu12 \
        nvidia-cusolver-cu12 \
        nvidia-cusparse-cu12 \
        nvidia-cusparselt-cu12 \
        nvidia-nccl-cu12 \
        nvidia-nvjitlink-cu12 \
        nvidia-nvshmem-cu12 \
        nvidia-nvtx-cu12 || true && \
    ${SYMFLUENCE_ENV}/bin/pip install --no-cache-dir --upgrade --force-reinstall \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.0.0,<3.0.0"

ENV BOOST_ROOT=${SYMFLUENCE_ENV}
ENV Boost_ROOT=${SYMFLUENCE_ENV}
ENV BOOST_INCLUDEDIR=${SYMFLUENCE_ENV}/include
ENV BOOST_LIBRARYDIR=${SYMFLUENCE_ENV}/lib
ENV Boost_NO_SYSTEM_PATHS=ON

RUN ${SYMFLUENCE_ENV}/bin/python -m ipykernel install \
    --prefix=/srv/conda/envs/notebook \
    --name "symfluence" \
    --display-name "Python (SYMFLUENCE)"

RUN CMAKE_PREFIX_PATH="${SYMFLUENCE_ENV}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}" \
    CPATH="${SYMFLUENCE_ENV}/include${CPATH:+:${CPATH}}" \
    LIBRARY_PATH="${SYMFLUENCE_ENV}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
    LD_LIBRARY_PATH="${SYMFLUENCE_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    LDFLAGS="${LDFLAGS:+${LDFLAGS} }-lexpat" \
    CMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:+${CMAKE_EXE_LINKER_FLAGS} }-lexpat" \
    ${SYMFLUENCE_ENV}/bin/symfluence binary install

RUN mamba install -n symfluence -y -c conda-forge \
    netcdf4 \
    h5py \
    hdf5 \
    h5netcdf \
    gdal

RUN conda clean -afy && mamba clean -afy

# Local build validation. Remove before pushing if you do not want build-time
# verification in the published Dockerfile.
RUN ${SYMFLUENCE_ENV}/bin/python -c "import h5py, netCDF4, h5netcdf; from osgeo import gdal; print('symfluence env validation ok')"

# ============================================================================
# END SYMFLUENCE INSTALLATION
# ============================================================================

RUN sed -i 's/\"default\": true/\"default\": false/g' /srv/conda/envs/notebook/share/jupyter/labextensions/@axlair/jupyterlab_vim/schemas/@axlair/jupyterlab_vim/plugin.json

USER ${NB_USER}

WORKDIR /home/${NB_USER}
