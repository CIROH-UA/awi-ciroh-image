# Start from a basic Ubuntu image
FROM ubuntu:20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set up environment variables
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    run-one \
    curl \
    gnupg2 \
    dbus-x11 \
    firefox \
    xfce4 \
    xfce4-panel \
    xfce4-session \
    xfce4-settings \
    xorg \
    xubuntu-icon-theme \
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
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    # The following line has been corrected
    $CONDA_DIR/bin/conda clean -a -y && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Set up conda environment
RUN conda update -n base -c conda-forge conda && \
    conda create -n notebook -c conda-forge \
    python=3.11 \
    jupyterlab \
    notebook \
    r-base \
    r-irkernel \
    r-devtools \
    r-sp \
    r-stringr \
    r-plyr \
    r-ggplot2 \
    r-terra \
    r-ncdf4 \
    nodejs \
    && conda clean --all -f -y

# Activate conda environment
ENV PATH /opt/conda/envs/notebook/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Install additional R packages
RUN source activate notebook && \
    R -e "install.packages(c('rgdal', 'ggmap'), repos='https://cloud.r-project.org/')" && \
    R -e "devtools::install_github('NCAR/rwrfhydro')"

# Install TurboVNC
ARG TURBOVNC_VERSION=2.2.6
RUN wget -q "https://sourceforge.net/projects/turbovnc/files/${TURBOVNC_VERSION}/turbovnc_${TURBOVNC_VERSION}_amd64.deb/download" -O turbovnc.deb && \
    apt-get update -qq --yes > /dev/null && \
    apt-get install -y ./turbovnc.deb > /dev/null && \
    apt-get remove -y light-locker > /dev/null && \
    rm ./turbovnc.deb && \
    ln -s /opt/TurboVNC/bin/* /usr/local/bin/ && \
    rm -rf /var/lib/apt/lists/*

# Install jupyter-remote-desktop-proxy
RUN pip install --no-cache-dir \
    https://github.com/jupyterhub/jupyter-remote-desktop-proxy/archive/main.zip

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

# Set up Jupyter Lab
RUN mkdir /opt/notebooks && \
    jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py

WORKDIR /opt/notebooks

# # Expose port for Jupyter Lab
# EXPOSE 8888

# # Start Jupyter Lab
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
USER ${NB_USER}
