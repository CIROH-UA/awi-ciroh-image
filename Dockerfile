FROM pangeo/pangeo-notebook:2024.04.08 AS base

ENV TROUTE_REPO=CIROH-UA/t-route
ENV TROUTE_BRANCH=ngiab
ENV NGEN_REPO=CIROH-UA/ngen
ENV NGEN_BRANCH=ngiab

USER root

# Install dependencies
RUN apt-get update && apt-get install -y \
    vim gfortran sqlite3 libsqlite3-dev \
    bzip2 libexpat1-dev libudunits2-dev zlib1g-dev \
    mpich libhdf5-dev libnetcdf-dev libnetcdff-dev libnetcdf-c++4-dev \
    sudo gcc-11 g++-11 make cmake ninja-build tar git gfortran \
    #python3.11 python3.11-dev python3-pip \
    flex bison wget

RUN mamba install -c conda-forge libboost -y
# Make RUN commands use the new environment
SHELL ["mamba", "run", "--no-capture-output", "-n", "notebook", "/bin/bash", "-c"]

###################################
FROM base AS troute_prebuild
WORKDIR /ngen
ENV FC=gfortran NETCDF=/usr/include
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install uv
ADD https://api.github.com/repos/${TROUTE_REPO}/git/refs/heads/${TROUTE_BRANCH} /tmp/version.json
RUN uv pip install --system -r https://raw.githubusercontent.com/$TROUTE_REPO/refs/heads/$TROUTE_BRANCH/requirements.txt
###################################
FROM troute_prebuild AS troute_build
WORKDIR /ngen/t-route
RUN git clone --depth 1 --single-branch --branch $TROUTE_BRANCH https://github.com/$TROUTE_REPO.git .
RUN git submodule update --init --depth 1
RUN uv pip install --system build wheel
RUN sed -i 's/build_[a-z]*=/#&/' compiler.sh
RUN ./compiler.sh no-e
#### TROUTE ####
RUN export CC=/usr/bin/gcc && \
    uv pip install --system --config-setting='--build-option=--use-cython' src/troute-network/
RUN uv build --wheel --config-setting='--build-option=--use-cython' src/troute-network/
RUN export CC=/usr/bin/gcc && \
    uv pip install --system --no-build-isolation --config-setting='--build-option=--use-cython' src/troute-routing/
RUN uv build --wheel --no-build-isolation --config-setting='--build-option=--use-cython' src/troute-routing/
RUN uv build --wheel --no-build-isolation src/troute-config/
RUN uv build --wheel --no-build-isolation src/troute-nwm/
###################################
FROM troute_prebuild AS ngen_clone
WORKDIR /ngen
ADD https://api.github.com/repos/${NGEN_REPO}/git/refs/heads/${NGEN_BRANCH} /tmp/version.json
RUN git clone --single-branch --branch $NGEN_BRANCH https://github.com/$NGEN_REPO.git && \
    cd ngen && \
    git submodule update --init --recursive --depth 1
##################################
FROM ngen_clone AS ngen_build
ENV PATH=/usr/bin:${PATH}:/usr/bin/mpich CC=/usr/bin/gcc
WORKDIR /ngen/ngen

ARG COMMON_BUILD_ARGS="-DNGEN_WITH_EXTERN_ALL=ON \
    -DNGEN_WITH_NETCDF:BOOL=ON \
    -DNGEN_WITH_BMI_C:BOOL=ON \
    -DNGEN_WITH_BMI_FORTRAN:BOOL=ON \
    -DNGEN_WITH_PYTHON:BOOL=ON \
    -DNGEN_WITH_ROUTING:BOOL=ON \
    -DNGEN_WITH_SQLITE:BOOL=ON \
    -DNGEN_WITH_UDUNITS:BOOL=ON \
    -DUDUNITS_QUIET:BOOL=ON \
    -DNGEN_WITH_TESTS:BOOL=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=. \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    "
SHELL ["mamba", "run", "--no-capture-output", "-n", "notebook", "/bin/bash", "-c"]
RUN cmake -G Ninja -B cmake_build_serial -S . ${COMMON_BUILD_ARGS} -DNGEN_WITH_MPI:BOOL=OFF && \
    cmake --build cmake_build_serial --target all -- -j $(nproc)

ARG MPI_BUILD_ARGS="-DNGEN_WITH_MPI:BOOL=ON \
    -DNetCDF_ROOT=/usr/lib/x86_64-linux-gnu \
    -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu"
RUN cmake -G Ninja -B cmake_build_parallel -S . ${COMMON_BUILD_ARGS} ${MPI_BUILD_ARGS} && \
    cmake --build cmake_build_parallel --target all -- -j $(nproc)

##################################
FROM ngen_build AS restructure_files
RUN mkdir -p /dmod/datasets /dmod/datasets/static /dmod/shared_libs /dmod/bin && \
    shopt -s globstar && \
    cp -a ./extern/**/cmake_build/*.so* /dmod/shared_libs/. || true && \
    cp -a ./extern/noah-owp-modular/**/*.TBL /dmod/datasets/static && \
    cp -a ./cmake_build_parallel/ngen /dmod/bin/ngen-parallel || true && \
    cp -a ./cmake_build_serial/ngen /dmod/bin/ngen-serial || true && \
    cp -a ./cmake_build_parallel/partitionGenerator /dmod/bin/partitionGenerator || true && \
    cd /dmod/bin && \
    (stat ngen-parallel && ln -s ngen-parallel ngen) || (stat ngen-serial && ln -s ngen-serial ngen)
###################################
# [LSTM-Update]
FROM base AS lstm_weights
RUN git clone --depth=1 --branch example_weights https://github.com/ciroh-ua/lstm.git /lstm_weights
# replace the relative path with the absolute path in the model config files
RUN shopt -s globstar
RUN sed -i 's|\.\.|/ngen/ngen/extern/lstm|g' /lstm_weights/trained_neuralhydrology_models/**/config.yml

###################################
FROM pangeo/pangeo-notebook:2024.04.08 AS final

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=${NB_PYTHON_PREFIX}/bin:$PATH

WORKDIR /ngen
COPY --from=ngen_build /ngen /ngen
COPY --from=restructure_files /dmod /dmod
COPY --from=troute_build /ngen/t-route/src/troute-*/dist/*.whl /tmp/
COPY --from=ngen_clone /ngen/ngen/extern/lstm/lstm /ngen/ngen/extern/lstm

#COPY --from=troute_build /tmp/troute_url /ngen/troute_url
#COPY --from=ngen_build /tmp/ngen_url /ngen/ngen_url

# Install runtime-only dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    mpich libnetcdf-dev libhdf5-dev libnetcdf-c++4-dev libudunits2-dev gnupg \
    #---------------------------------------------
    # 2i2c: Packages for Linux Desktop
    #---------------------------------------------
    xfce4 xfce4-terminal tigervnc-standalone-server \
    x11vnc supervisor wget curl ca-certificates \
    xdg-utils libgtk-3-0 libdbus-glib-1-2 libx11-xcb1 libnss3 libxss1 libasound2 \
    libxcomposite1 libxdamage1 libxrandr2 libxt6 libatk1.0-0 libpango-1.0-0 \
    libgdk-pixbuf2.0-0 fonts-liberation \
    #---------------------------------------------
    # TEEHR: (https://rtiinternational.github.io/teehr/getting_started/index.html)
    #---------------------------------------------
    openjdk-17-jdk \
    #---------------------------------------------
    # 2i2c: Google Cloud SDK (gcloud, gsutil)
    #---------------------------------------------
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - \
    && apt-get update -y \
    && apt-get install google-cloud-sdk -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set environment for ngen
RUN ln -s /dmod/bin/ngen /usr/local/bin/ngen
ENV FC=gfortran NETCDF=/usr/include PATH=$PATH:/usr/bin/mpich
# [LSTM-Update]
ENV UV_COMPILE_BYTECODE=1

# Set softlink for mpi (required for spotpy calibration)
#RUN ln -s /usr/lib/x86_64-linux-gnu/libmpi.so /usr/lib/x86_64-linux-gnu/libmpi.so.12

# Install firefox for interactive workflows
RUN mkdir -p /opt/firefox && \
    wget -O /tmp/firefox.tar.bz2 "https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US" && \
    tar -xf /tmp/firefox.tar.bz2 -C /opt/firefox --strip-components=1 && \
    ln -s /opt/firefox/firefox /usr/local/bin/firefox && \
    rm /tmp/firefox.tar.bz2

# Set Firefox as the default browser
ENV BROWSER=/usr/local/bin/firefox
ENV XDG_BROWSER=/usr/local/bin/firefox

# Set environment variables for TEEHR
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

RUN pip3 install uv && \
    uv pip install --system --no-cache-dir \
    numpy==$(/dmod/bin/ngen --info | grep -m 1 -e 'NumPy Version: ' | cut -d ':' -f 2 | uniq | xargs) \
    jupyterlab_vim \
    teehr==0.4.* \
    git-lfs==1.6 \
    #---------------------------------------------
    # 2i2c: Install GIS packages
    #---------------------------------------------
    spatialpandas \
    easydev \
    colormap \
    colorcet \
    duckdb \
    dask_geopandas \
    hydrotools \
    sidecar \
    dataretrieval \
    google-cloud-bigquery \
    #---------------------------------------------
    # 2i2c: To enable linux desktop
    #---------------------------------------------
    jupyter-remote-desktop-proxy \
    websockify \
    #---------------------------------------------
    # 2i2c: Hydroshare & teehr packages
    #---------------------------------------------
    git+https://github.com/hydroshare/nbfetch.git@hspuller-auth \
    dask_labextension \
    hsfiles-jupyter \
    #---------------------------------------------
    # Ngen: calibration spotpy
    #---------------------------------------------
    spotpy \
    # mpi4py \
    # ipyparallel \
    #---------------------------------------------
    # 2i2c: To enable venv kernels in Jupyter
    #---------------------------------------------
    #ipykernel
    #---------------------------------------------
    # Misc:
    #   - TEEHR: Download the required JAR files for Spark to interact with AWS S3.
    #   - Link hsfiles-jupyter to JupyterLab
    #---------------------------------------------
    && uv run python -m teehr.utils.install_spark_jars \
    && uv run python -m hsfiles_jupyter

RUN echo "/dmod/shared_libs/" >> /etc/ld.so.conf.d/ngen.conf && ldconfig -v

# Upgrade colorama to resolve dependency conflict
RUN uv pip install --system --upgrade colorama

# Install nb_black separately to address metadata generation issue
RUN uv pip install --system --no-cache-dir nb_black==1.0.5

# enable jupyter_server extension
RUN jupyter server extension enable --py nbfetch --sys-prefix

# Update custom Jupyter Lab settings
RUN sed -i 's/\"default\": true/\"default\": false/g' /srv/conda/envs/notebook/share/jupyter/labextensions/@axlair/jupyterlab_vim/schemas/@axlair/jupyterlab_vim/plugin.json

##########
# While creating a venv inside docker is not a good idea, some packages required
# by 2i2c and hydroshare (nbfetch) require pydantic>1 and numpy latest version.
# At the same time, the routing module of ngen is built with pydantic1 and a
# specific version of numpy.
# In order for ngen to work with 2i2c and hydroshare packages, conflicting packages
# are installed in a venv which will be referenced by the PyNGIAB package
#
# WARN: Everything installed after this using `uv` will be installed in the venv
##########
RUN uv venv --system-site-packages \
    # To avoid issues with installing lstm from seperate pip index
    && uv pip install --no-cache-dir \
          /ngen/ngen/extern/lstm --extra-index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --no-cache-dir \
    /tmp/*.whl \
    'netCDF4>=1.6.5' \
    numpy==$(/dmod/bin/ngen --info | grep -m 1 -e 'NumPy Version: ' | cut -d ':' -f 2 | uniq | xargs) \
    'pydantic<2' \
    #---------------------------------------------
    # Ngen: calibration ngen-cal
    #---------------------------------------------
    "git+https://github.com/noaa-owp/ngen-cal@master#egg=ngen_cal&subdirectory=python/ngen_cal" \
    #---------------------------------------------
    # Setup and install ngiab_data_preprocess module to allow preparing data for ngiab
    #   - [Optional] Download default hydrofabric for ngiab_data_preprocess
    #---------------------------------------------
    ngiab_data_preprocess==4.6.5 \
    #&& uv run python -c "from data_sources.source_validation import download_and_update_hf; \
    #			 download_and_update_hf();" \
    && rm -rf /tmp/*.whl

# [LSTM-Update] Replace the noaa-owp example weights with jmframes
RUN rm -rf /ngen/ngen/extern/lstm/trained_neuralhydrology_models
COPY --from=lstm_weights /lstm_weights/trained_neuralhydrology_models /ngen/ngen/extern/lstm/trained_neuralhydrology_models

# Make this venv available as JupyterHub kernel
# ENV PATH=/ngen/.venv/bin:$PATH
# RUN python -m ipykernel install --name=ngiab-pydantic1
# #RUN python -m ipykernel install --user --name=NGIAB

# To avoid error for ngen-parallel
ENV RDMAV_FORK_SAFE=1

##########
# PyNGIAB (https://github.com/fbaig/ciroh_pyngiab)
##########
RUN pip install git+https://github.com/fbaig/ciroh_pyngiab.git

COPY ./tests /tests

#USER root
# Update permissions to allow Jupyter non-root user to install and use packages
RUN chown -R ${NB_USER}:${NB_USER} \
    /home/jovyan/ \
    /tests/ \
    #/home/jovyan/.ngiab/ \
    #/srv/conda/ \
    && chmod +x /tests/test-entrypoint.sh

USER ${NB_USER}
WORKDIR /ngen/
RUN echo "export PS1='\u\[\033[01;32m\]@ngiab_dev\[\033[00m\]:\[\033[01;35m\]\W\[\033[00m\]\$ '" >> ~/.bashrc
# # Download hydrofabric when starting container
# ENTRYPOINT uv run python -c "from data_sources.source_validation import download_and_update_hf; download_and_update_hf();"
