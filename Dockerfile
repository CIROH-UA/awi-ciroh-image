# Headless AWI CIROH image with SYMFLUENCE.
# This variant keeps the current notebook + SYMFLUENCE workflow but removes
# desktop/VNC components and Google Cloud SDK to reduce image size.

FROM pangeo/pangeo-notebook:2024.04.08

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=${NB_PYTHON_PREFIX}/bin:$PATH

# Install build dependencies and runtime libraries in single layer to minimize bloat
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    gfortran \
    make \
    cmake \
    pkg-config \
    libopenmpi-dev \
    openmpi-bin \
    libgdal-dev \
    gdal-bin \
    libhdf5-dev \
    libhdf5-openmpi-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    libblas-dev \
    libblas3 \
    liblapack-dev \
    liblapack3 \
    libopenblas-dev \
    libopenblas0-pthread \
    libproj-dev \
    libgeos-dev \
    libudunits2-dev \
    libexpat1-dev \
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

# SYMFLUENCE uses this env variable for installing the tools (e.g. SUMMA)
ENV SYMFLUENCE_DATA_DIR=/opt/symfluence/data
ENV SYMFLUENCE_CODE_DIR=/opt/symfluence
ENV SYMFLUENCE_ENV=/srv/conda/envs/symfluence

RUN mkdir -p /opt/symfluence/data && chmod -R 755 /opt/symfluence

# Keep compiled numeric stack conda-managed to avoid ABI conflicts.
# Group conda/mamba installs to reduce layer count.
RUN mamba install -n ${CONDA_ENV} -y -c conda-forge \
    websockify \
    "numpy<2" \
    pandas \
    pyarrow && \
    mamba clean -afy

# Create SYMFLUENCE environment and upgrade pip/setuptools/wheel in one layer
RUN mamba create -n symfluence -y -c conda-forge \
        python=3.11 \
        ipykernel \
        "boost-cpp>=1.79" \
        pip && \
    ${SYMFLUENCE_ENV}/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    mamba clean -afy

# Install notebook-focused Python packages (base env)
# Separate layer to enable cache reuse if notebook packages change
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
    dataretrieval \
    hsfiles-jupyter \
    colormap \
    hsclient[all]==1.1.6 \
    pydantic==2.7.* \
    colorama

# Install git+https packages separately for better cache locality
RUN pip install -U --no-cache-dir --upgrade-strategy only-if-needed \
    git+https://github.com/hydroshare/nbfetch.git@v0.6.4

RUN jupyter server extension enable --py nbfetch --sys-prefix && \
    python -m hsfiles_jupyter

# ============================================================================
# SYMFLUENCE INSTALLATION SECTION
# ============================================================================

# Network can be flaky when pip resolves/downloads SYMFLUENCE's large dependency set.
# Retry a few times to avoid transient IncompleteRead/ProtocolError failures.
RUN set -eux; \
    success=0; \
    for attempt in 1 2 3; do \
      if ${SYMFLUENCE_ENV}/bin/python -m pip install \
        --no-cache-dir \
        --prefer-binary \
        --progress-bar off \
        --retries 20 \
        --timeout 120 \
        -v \
        git+https://github.com/DarriEy/SYMFLUENCE.git@v0.9.2; then \
        success=1; \
        break; \
      fi; \
      echo "SYMFLUENCE install attempt ${attempt} failed; retrying..."; \
      sleep 15; \
    done; \
    test "$success" -eq 1

# Validate SYMFLUENCE installation
RUN ${SYMFLUENCE_ENV}/bin/python -m pip show symfluence && \
    ${SYMFLUENCE_ENV}/bin/python -m symfluence.main_cli --version

# Remove optional GPU wheels that may appear as transitive deps
RUN ${SYMFLUENCE_ENV}/bin/pip uninstall -y \
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
        nvidia-nvtx-cu12 || true

# Install CPU-only PyTorch
RUN ${SYMFLUENCE_ENV}/bin/pip install --no-cache-dir --upgrade --force-reinstall \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.0.0,<3.0.0"

# Install CPU-only JAX and the jax-related extras into the symfluence environment.
# We install these packages directly (rather than using the symfluence[jax]
# extra) to avoid re-installing or replacing the symfluence package itself.
# Use the official jax release index for CPU wheels.
# Here the symfluence jax dependencies are pinned: https://github.com/symfluence-org/SYMFLUENCE/blob/main/pyproject.toml
# NOTE: Update the pinned versions in the Dockerfile as needed to match the symfluence pyproject.toml.
RUN set -eux; \
    success=0; \
    for attempt in 1 2 3; do \
      if ${SYMFLUENCE_ENV}/bin/pip install --no-cache-dir --upgrade \
         "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html \
         "jsnow17>=0.1.0" "jsacsma>=0.2.3" "jxaj>=0.2.3" "jhbv>=0.2.4" "jhechms>=0.2.3" "jtopmodel>=0.2.3"; then \
        success=1; \
        break; \
      fi; \
      echo "JAX+extras install attempt ${attempt} failed; retrying..."; \
      sleep 15; \
    done; \
    test "$success" -eq 1

ENV BOOST_ROOT=${SYMFLUENCE_ENV}
ENV Boost_ROOT=${SYMFLUENCE_ENV}
ENV BOOST_INCLUDEDIR=${SYMFLUENCE_ENV}/include
ENV BOOST_LIBRARYDIR=${SYMFLUENCE_ENV}/lib
ENV Boost_NO_SYSTEM_PATHS=ON

# Install kernel and configure environment
RUN ${SYMFLUENCE_ENV}/bin/python -m ipykernel install \
    --prefix=/srv/conda/envs/notebook \
    --name "symfluence" \
    --display-name "Python (SYMFLUENCE)" && \
    ${SYMFLUENCE_ENV}/bin/python -c 'import json; from pathlib import Path; kernel_path = Path("/srv/conda/envs/notebook/share/jupyter/kernels/symfluence/kernel.json"); kernel_spec = json.loads(kernel_path.read_text()); kernel_spec["env"] = {**kernel_spec.get("env", {}), "PATH": "/srv/conda/envs/symfluence/bin:${PATH}", "CONDA_PREFIX": "/srv/conda/envs/symfluence"}; kernel_path.write_text(json.dumps(kernel_spec, indent=2) + "\n")'

# Build SYMFLUENCE binaries with optimized build environment
RUN CMAKE_PREFIX_PATH="${SYMFLUENCE_ENV}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}" \
    CPATH="${SYMFLUENCE_ENV}/include${CPATH:+:${CPATH}}" \
    LIBRARY_PATH="${SYMFLUENCE_ENV}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
    LD_LIBRARY_PATH="${SYMFLUENCE_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    LDFLAGS="${LDFLAGS:+${LDFLAGS} }-lexpat" \
    CMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:+${CMAKE_EXE_LINKER_FLAGS} }-lexpat" \
    SYMFLUENCE_PYTHON="${SYMFLUENCE_ENV}/bin/python" \
    ${SYMFLUENCE_ENV}/bin/symfluence binary install

# Replace pip-installed HDF5/NetCDF bindings with conda equivalents in same layer
# to avoid carrying both copies in the final image
RUN ${SYMFLUENCE_ENV}/bin/pip uninstall -y \
    h5py \
    netCDF4 \
    h5netcdf || true && \
    mamba install -n symfluence -y -c conda-forge \
    netcdf4 \
    h5py \
    hdf5 \
    h5netcdf \
    gdal && \
    conda clean -afy && mamba clean -afy

# Local build validation
RUN ${SYMFLUENCE_ENV}/bin/python -c "import h5py, netCDF4, h5netcdf; from osgeo import gdal; print('symfluence env validation ok')"

# ============================================================================
# END SYMFLUENCE INSTALLATION
# ============================================================================

RUN sed -i 's/"default": true/"default": false/g' /srv/conda/envs/notebook/share/jupyter/labextensions/@axlair/jupyterlab_vim/schemas/@axlair/jupyterlab_vim/plugin.json

# This symlink must be created at runtime because /home/${NB_USER} is a
# mounted volume in Kubernetes and any build-time writes to it are shadowed.
# Scripts placed in before-notebook.d/ are sourced by start-notebook.sh
# before the Jupyter server starts.
RUN mkdir -p /usr/local/bin/before-notebook.d && \
    printf '#!/bin/bash\nmkdir -p /home/${NB_USER}/SYMFLUENCE_data\nln -sfn %s/installs /home/${NB_USER}/SYMFLUENCE_data/installs\n' \
    "${SYMFLUENCE_DATA_DIR}" \
    > /usr/local/bin/before-notebook.d/50-symfluence-symlink.sh && \
    chmod +x /usr/local/bin/before-notebook.d/50-symfluence-symlink.sh

# SYMFLUENCE expects this env variable to be set for finding the tools (e.g. SUMMA) at runtime.
ENV SYMFLUENCE_DATA_DIR=/home/${NB_USER}/SYMFLUENCE_data

USER ${NB_USER}

WORKDIR /home/${NB_USER}
