# This Dockerfile aims to provide a Pangeo-style image with the R environment
# It was constructed by modifying the original to focus solely on R support.

# Use a base image with minimal Python packages
FROM pangeo/pangeo-notebook:2024.04.08

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH ${NB_PYTHON_PREFIX}/bin:$PATH

# Install necessary packages
RUN apt-get update -qq --yes && \
    apt-get install -y --no-install-recommends \
    gnupg2 \
    dbus-x11 \
    firefox \
    xfce4 \
    xfce4-panel \
    xfce4-session \
    xfce4-settings \
    xorg \
    xubuntu-icon-theme \
    curl \
    libfontconfig1-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    nco \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libudunits2-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm (if needed for any web-based tools)
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

# Install jupyter-remote-desktop-proxy if still needed
RUN export PATH=${NB_PYTHON_PREFIX}/bin:${PATH} \
 && npm install -g npm@7.24.0 \
 && pip install --no-cache-dir \
        https://github.com/jupyterhub/jupyter-remote-desktop-proxy/archive/main.zip

# Install Google Cloud SDK (gcloud, gsutil)
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

# Install R and IRkernel
RUN mamba install -n ${CONDA_ENV} -c conda-forge r-base r-irkernel r-devtools r-sp r-stringr r-plyr r-ggplot2 r-terra r-ncdf4 -y && \
    /srv/conda/envs/notebook/bin/R -e "IRkernel::installspec(user = FALSE)"

# Install additional R packages
RUN /srv/conda/envs/notebook/bin/R -e "install.packages(c('rgdal', 'ggmap'), repos='https://cloud.r-project.org/')" && \
    /srv/conda/envs/notebook/bin/R -e "devtools::install_github('NCAR/rwrfhydro')"


# # Install tini
# RUN apt-get update && apt-get install -y tini

# # Set environment variables
# ENV JUPYTER_ENABLE_LAB=yes

# # Set the entrypoint to start Jupyter Lab
# ENTRYPOINT ["tini", "-g", "--"]
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

USER ${NB_USER}
