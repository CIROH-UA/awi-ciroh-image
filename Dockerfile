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
    flex bison wget curl \
    #---------------------------------------------
    # Extern models ported from NGIAB-CloudInfra: SUMMA needs OpenBLAS
    #---------------------------------------------
    libopenblas-dev

RUN mamba install -c conda-forge libboost -y
# Compiler env shared by ngen, t-route, and the extern model build stages
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++ FC=gfortran
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
# Record the exact commit built, for provenance in the final image
RUN echo $(git remote get-url origin | sed 's/\.git$//' | awk '{print $0 "/tree/" }' | tr -d '\n' && git rev-parse HEAD) >> /tmp/troute_url
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
# Record the exact commit built, for provenance in the final image
RUN echo $(git remote get-url origin | sed 's/\.git$//' | awk '{print $0 "/tree/" }' | tr -d '\n' && git rev-parse HEAD) >> /tmp/ngen_url

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
# [Extern models ported from NGIAB-CloudInfra]
FROM ngen_clone AS build_sundials
WORKDIR /sundials
ENV SUNDIALS_VERSION=7.5.0
RUN wget https://github.com/LLNL/sundials/releases/download/v${SUNDIALS_VERSION}/sundials-${SUNDIALS_VERSION}.tar.gz && \
    tar -xzf sundials-${SUNDIALS_VERSION}.tar.gz && \
    rm sundials-${SUNDIALS_VERSION}.tar.gz
RUN cmake -G Ninja -B build_sundials sundials-${SUNDIALS_VERSION} \
    -DEXAMPLES_ENABLE_C=OFF -DEXAMPLES_ENABLE_F2003=OFF \
    -DBUILD_FORTRAN_MODULE_INTERFACE=ON -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_INSTALL_PREFIX=/sundials/install && \
    cmake --build build_sundials --target all -- -j $(nproc) && \
    cmake --build build_sundials --target install
###################################
FROM build_sundials AS build_summa
WORKDIR /ngen/ngen/extern/summa
RUN cmake -G Ninja -B build_summa -DUSE_NEXTGEN=ON -DUSE_SUNDIALS=ON \
    -DSPECIFY_LAPACK_LINKS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DNetCDF_F90_INCLUDE_DIR=/usr/include \
    -DOpenBLAS_INCLUDE_DIR=/usr/include \
    -DSUNDIALS_DIR=/sundials/build_sundials/ && \
    cmake --build build_summa --target all -- -j $(nproc)
###################################
FROM ngen_clone AS build_sacsma
WORKDIR /ngen/ngen/extern/sac-sma
RUN cmake -B cmake_build -DISO_C_FORTRAN_BMI_PATH=/ngen/ngen/extern/iso_c_fortran_bmi \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_Fortran_COMPILER=gfortran -S . && \
    cmake --build cmake_build -j $(nproc)
###################################
FROM ngen_clone AS build_snow17
WORKDIR /ngen/ngen/extern/snow17
RUN cmake -B cmake_build -DISO_C_FORTRAN_BMI_PATH=/ngen/ngen/extern/iso_c_fortran_bmi \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_Fortran_COMPILER=gfortran -S . && \
    cmake --build cmake_build -j $(nproc)
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
COPY --from=build_summa /ngen/ngen/extern/summa/build_summa/*.so /dmod/shared_libs/
COPY --from=build_sacsma /ngen/ngen/extern/sac-sma/cmake_build/*.so /dmod/shared_libs/
COPY --from=build_snow17 /ngen/ngen/extern/snow17/cmake_build/*.so /dmod/shared_libs/
###################################
# [LSTM-Update]
FROM base AS lstm_weights
# uv/rust-lstm-1025's convert.py have no dependency on the conda "notebook" env
SHELL ["/bin/bash", "-c"]
RUN git clone --depth=1 --branch example_weights https://github.com/ciroh-ua/lstm.git /lstm_weights
# uv is needed to run the rust-lstm-1025 weight conversion script below
ENV UV_INSTALL_DIR=/root/.cargo/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"
# Convert the example weights to the format expected by the rust LSTM (librust_lstm_1025.so)
RUN uv run --with pyyaml --with numpy --with torch --extra-index-url https://download.pytorch.org/whl/cpu \
    https://raw.githubusercontent.com/CIROH-UA/rust-lstm-1025/refs/tags/v0.1.0/scripts/convert.py \
    all /lstm_weights/trained_neuralhydrology_models/
# replace the relative path with the absolute path in the model config files
RUN shopt -s globstar
RUN sed -i 's|\.\.|/ngen/ngen/extern/lstm|g' /lstm_weights/trained_neuralhydrology_models/**/config.yml
###################################
# [Rust LSTM ported from NGIAB-CloudInfra]
FROM base AS burn_lstm
# cargo/rustc have no dependency on the conda "notebook" env
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y clang && rm -rf /var/lib/apt/lists/*
# Pin the install location explicitly rather than relying on $HOME (which the
# base Jupyter image may point somewhere other than /root even while USER root)
ENV CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
WORKDIR /build
RUN git clone --depth=1 --branch v0.1.2 https://github.com/ciroh-ua/rust-lstm-1025
WORKDIR /build/rust-lstm-1025
# PATH (inherited from the base Jupyter image) puts the conda env's own
# cross-compiler ahead of /usr/bin, so bare `cc` resolves to conda's gcc
# instead of the system one, which lacks a compatible libgcc for linking.
# Force the actual system gcc explicitly.
ENV RUSTFLAGS="-C linker=/usr/bin/gcc"
RUN cargo build --release

###################################
FROM pangeo/pangeo-notebook:2024.04.08 AS final
# Packages ngen's routing module (and the extern models built above) require to
# stay pinned even as later pip/uv installs pull in other Python dependencies.
ARG pinned_python_packages="netCDF4>=1.6.5 pydantic<2 pandas>=2.0,<3.0"

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=${NB_PYTHON_PREFIX}/bin:$PATH

WORKDIR /ngen
COPY --from=ngen_build /ngen /ngen
COPY --from=restructure_files /dmod /dmod
COPY --from=troute_build /ngen/t-route/src/troute-*/dist/*.whl /tmp/
COPY --from=ngen_clone /ngen/ngen/extern/lstm/lstm /ngen/ngen/extern/lstm
COPY --from=burn_lstm /build/rust-lstm-1025/target/release/librust_lstm_1025.so /dmod/shared_libs/librust_lstm_1025.so
COPY --from=build_sundials /sundials/install /sundials

COPY --from=troute_build /tmp/troute_url /ngen/troute_url
COPY --from=ngen_build /tmp/ngen_url /ngen/ngen_url

# Install runtime-only dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    mpich libnetcdf-dev libhdf5-dev libnetcdf-c++4-dev libudunits2-dev gnupg \
    #---------------------------------------------
    # Extern models (SUMMA/SUNDIALS) ported from NGIAB-CloudInfra
    #---------------------------------------------
    libopenblas-dev libnetcdff-dev \
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
    && apt-get install google-cloud-cli -y --no-install-recommends \
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
    teehr==0.5.* \
    git-lfs==1.6 \
    #---------------------------------------------
    # 2i2c: Install GIS packages
    #---------------------------------------------
    dask==2025.12.0 distributed==2025.12.0 \
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
    #&& uv run python -m teehr.utils.install_spark_jars \
    && uv run python -m hsfiles_jupyter

RUN echo "/dmod/shared_libs/" >> /etc/ld.so.conf.d/ngen.conf && \
    echo "/sundials/lib" >> /etc/ld.so.conf.d/sundials.conf && \
    echo "/sundials/lib64" >> /etc/ld.so.conf.d/sundials.conf && \
    ldconfig -v

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
    ngiab_data_preprocess==4.6.7 \
    'pandas>=2.0,<3.0' \
    #&& uv run python -c "from data_sources.source_validation import download_and_update_hf; \
    #			 download_and_update_hf();" \
    && rm -rf /tmp/*.whl

# [dHBV2] MHPI dHBV2 model, ported from NGIAB-CloudInfra (installed into the
# same ngen venv created above)
RUN uv pip install --no-cache-dir \
    "dmg==1.4.3" "hydrodl2==1.3.5" "dhbv2==0.5.4" \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN mkdir -p /ngen/ngen/extern/dhbv2/ngen_resources/data/dhbv_2_mts/model/dhbv_2_mts/ \
            /ngen/ngen/extern/dhbv2/ngen_resources/data/dhbv_2/model/dhbv_2/ && \
    curl -fsSL https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/models/owp/dhbv_2_mts.tar.gz \
        | tar -xz -C /ngen/ngen/extern/dhbv2/ngen_resources/data/dhbv_2_mts/model/dhbv_2_mts/ --strip-components=1 && \
    curl -fsSL https://mhpi-spatial.s3.us-east-2.amazonaws.com/mhpi-release/models/owp/dhbv_2.tar.gz \
        | tar -xz -C /ngen/ngen/extern/dhbv2/ngen_resources/data/dhbv_2/model/dhbv_2/ --strip-components=1

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

# Defensive re-assertion: guard against any of the installs above (dhbv2/hydrodl2,
# ngiab_data_preprocess, PyNGIAB) silently upgrading a package ngen's routing
# module depends on.
RUN uv pip install --no-cache-dir ${pinned_python_packages}

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
