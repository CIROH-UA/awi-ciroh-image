FROM pangeo/pangeo-notebook:2024.04.08

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=${NB_PYTHON_PREFIX}/bin:$PATH
ENV NB_USER=jovyan
ENV HOME=/home/${NB_USER}

# Update and install basic utilities and R dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    xz-utils \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libudunits2-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libxt-dev \
    nco \
    proj-data \
    dbus-x11 \
    firefox \
    xfce4 \
    xfce4-panel \
    xfce4-session \
    xfce4-settings \
    xorg \
    xubuntu-icon-theme \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Basic utilities and R dependencies installed successfully"

# Install Node.js
ENV NODE_VERSION=16.20.0
RUN curl -fsSLO --compressed "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz" \
    && tar -xJf "node-v$NODE_VERSION-linux-x64.tar.xz" -C /usr/local --strip-components=1 --no-same-owner \
    && rm "node-v$NODE_VERSION-linux-x64.tar.xz" \
    && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
    && echo "Node.js installed successfully"

# Install TurboVNC
ARG TURBOVNC_VERSION=2.2.6
RUN wget -q "https://sourceforge.net/projects/turbovnc/files/${TURBOVNC_VERSION}/turbovnc_${TURBOVNC_VERSION}_amd64.deb/download" -O turbovnc.deb \
    && apt-get update -qq --yes > /dev/null \
    && apt-get install -y ./turbovnc.deb > /dev/null \
    && apt-get remove -y light-locker > /dev/null \
    && rm ./turbovnc.deb \
    && ln -s /opt/TurboVNC/bin/* /usr/local/bin/ \
    && rm -rf /var/lib/apt/lists/* \
    && echo "TurboVNC installed successfully"

# Set up permissions for jovyan user
RUN mkdir -p ${HOME}/.local/share/jupyter \
    && chown -R ${NB_USER}:users ${HOME}

# Switch to jovyan user for conda and pip installations
USER ${NB_USER}
WORKDIR ${HOME}

# Ensure conda and npm are in the PATH
ENV PATH=${HOME}/.local/bin:${CONDA_DIR}/bin:$PATH

# Install websockify
RUN conda install -n ${CONDA_ENV} -y websockify && \
    echo "websockify installed successfully" || echo "websockify installation failed"

# Update npm
RUN npm install -g npm@7.24.0 && \
    echo "npm updated successfully" || echo "npm update failed"

# Install jupyter-remote-desktop-proxy
RUN pip install --no-cache-dir \
    https://github.com/jupyterhub/jupyter-remote-desktop-proxy/archive/main.zip && \
    echo "jupyter-remote-desktop-proxy installed successfully" || echo "jupyter-remote-desktop-proxy installation failed"

# Rest of the Dockerfile remains the same...

# Install R, R kernel, and R packages using conda
RUN conda install -n ${CONDA_ENV} -c conda-forge \
    r-base \
    r-irkernel \
    r-essentials \
    r-devtools \
    r-rgdal \
    r-stringr \
    r-plyr \
    r-ggplot2 \
    r-ggmap \
    r-terra \
    r-ncdf4 \
    && echo "R, R kernel, and R packages installed via conda"

# Install rwrfhydro from GitHub
RUN R -e "devtools::install_github('NCAR/rwrfhydro')" \
    && echo "rwrfhydro installed"

# Install additional Python packages and Jupyter extensions
RUN pip install --no-cache-dir \
    jupyter-tree-download \
    spatialpandas \
    easydev \
    colormap \
    colorcet \
    duckdb \
    dask_geopandas \
    hydrotools \
    sidecar \
    google-cloud-bigquery \
    dataretrieval \
    nb_black==1.0.5 \
    git+https://github.com/hydroshare/nbfetch.git@hspuller-auth && \
    echo "Python packages and Jupyter extensions installed successfully" || echo "Some Python packages failed to install"

# Install jupyterlab_vim separately
RUN pip install --no-cache-dir jupyterlab_vim && \
    jupyter labextension install @axlair/jupyterlab_vim && \
    echo "jupyterlab_vim installed successfully" || echo "jupyterlab_vim installation failed"


# Enable jupyter_server extension
RUN jupyter server extension enable --py nbfetch --sys-prefix \
    && echo "jupyter_server extension enabled successfully"

# Switch back to root for Google Cloud SDK installation
USER root

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update -y \
    && apt-get install -y google-cloud-sdk \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Google Cloud SDK installed successfully"

# Update custom Jupyter Lab settings and run diagnostics
RUN echo "Checking Jupyter Lab extensions..." && \
    jupyter labextension list || true && \
    echo "Checking for vim plugin file..." && \
    find /srv/conda/envs/notebook/share/jupyter -name "plugin.json" | grep jupyterlab_vim || true && \
    if [ -f /srv/conda/envs/notebook/share/jupyter/labextensions/@axlair/jupyterlab_vim/schemas/@axlair/jupyterlab_vim/plugin.json ]; then \
        sed -i 's/\"default\": true/\"default\": false/g' /srv/conda/envs/notebook/share/jupyter/labextensions/@axlair/jupyterlab_vim/schemas/@axlair/jupyterlab_vim/plugin.json && \
        echo "Jupyter Lab settings updated successfully"; \
    else \
        echo "Jupyter Lab vim plugin file not found, skipping settings update"; \
    fi && \
    echo "Jupyter Lab settings check completed"

# Switch back to jovyan user
USER ${NB_USER}

# # Verify R installation and packages
# RUN R --version && \
#     R -e "installed.packages()[,c(1,3:4)]" && \
#     echo "R installation and packages verified"

# # Verify conda environment
# RUN conda list -n ${CONDA_ENV} && \
#     echo "Conda environment verified"

# # Verify Jupyter kernels
# RUN jupyter kernelspec list && \
#     echo "Jupyter kernels listed"

# # Verify R kernel installation
# RUN R -e "IRkernel::installspec(user = TRUE)" \
#     && jupyter kernelspec list \
#     && echo "R kernel installation verified"

# # Set environment variables
# ENV JUPYTER_ENABLE_LAB=yes

# # Set the entrypoint to start Jupyter Lab
# ENTRYPOINT ["tini", "-g", "--"]
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
