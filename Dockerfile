# ============================================================================
# STAGE 1: SYMFLUENCE builder
# ============================================================================
FROM pangeo/pangeo-notebook:2024.04.08 AS symfluence-builder

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ gfortran cmake \
        git wget curl \
        libopenmpi-dev openmpi-bin \
        libgdal-dev gdal-bin \
        libhdf5-dev libhdf5-openmpi-dev \
        libnetcdf-dev libnetcdff-dev \
        libblas-dev liblapack-dev libopenblas-dev \
        libproj-dev libgeos-dev \
        cdo \
    && rm -rf /var/lib/apt/lists/*

# Set SYMFLUENCE data path
ENV SYMFLUENCE_DATA_DIR=/opt/symfluence/data
ENV SYMFLUENCE_CODE_DIR=/opt/symfluence

RUN mkdir -p /opt/symfluence/data && chmod -R 755 /opt/symfluence

# Create isolated conda env for SYMFLUENCE
RUN mamba create -n symfluence -y python=3.11 ipykernel

# Install SYMFLUENCE and binaries
RUN /srv/conda/envs/symfluence/bin/pip install \
        git+https://github.com/DarriEy/SYMFLUENCE.git@main

RUN /srv/conda/envs/symfluence/bin/symfluence binary install

# Clean up conda caches & reduce environment size
RUN conda clean -afy && \
    find /srv/conda/envs/symfluence/lib/python3.11/site-packages/ -type d -name "tests" -exec rm -rf {} + && \
    find /srv/conda/envs/symfluence/lib/python3.11/site-packages/ -type d -name "doc" -exec rm -rf {} + && \
    find /srv/conda/envs/symfluence/lib/python3.11/site-packages/ -name "*.pyc" -delete

# ============================================================================
# STAGE 2: Final image
# ============================================================================
FROM pangeo/pangeo-notebook:2024.04.08

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/srv/conda/envs/symfluence/bin:$PATH

# Install only runtime libraries needed for SYMFLUENCE
RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin \
        gdal-bin \
        libhdf5-103 \
        libnetcdf19 \
        libproj25 \
        libgeos-c1v5 \
        cdo \
    && rm -rf /var/lib/apt/lists/*

# Copy SYMFLUENCE conda environment & binaries from builder
COPY --from=symfluence-builder /srv/conda/envs/symfluence \
     /srv/conda/envs/symfluence

COPY --from=symfluence-builder /opt/symfluence \
     /opt/symfluence

ENV SYMFLUENCE_DATA_DIR=/opt/symfluence/data

# Register SYMFLUENCE kernel in notebook env
RUN /srv/conda/envs/symfluence/bin/python -m ipykernel install \
        --prefix=/srv/conda/envs/notebook \
        --name symfluence \
        --display-name "Python (SYMFLUENCE)"

# --------------------------------------------------------------------------
# Install Python packages for notebooks
# --------------------------------------------------------------------------
RUN mamba install -n ${CONDA_ENV} -y -c conda-forge \
        "numpy<2" pandas pyarrow websockify

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
        google-cloud-bigquery \
        dataretrieval \
        hsfiles-jupyter \
        nb_black==1.0.5

RUN pip install --no-cache-dir --no-deps colormap
RUN pip install --upgrade colorama

# HydroShare packages
RUN pip install --no-cache-dir \
        hsclient[all]==1.1.6 \
        "pydantic==2.7.*" && \
    pip install --no-cache-dir \
        git+https://github.com/hydroshare/nbfetch.git@v0.6.4 && \
    jupyter server extension enable --py nbfetch --sys-prefix && \
    python -m hsfiles_jupyter

# Switch back to notebook user
USER ${NB_USER}

WORKDIR /home/${NB_USER}
