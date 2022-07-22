# This Dockerfile aims to provide a Pangeo-style image with the VNC/Linux Desktop feature
# It was constructed by following the instructions and copying code snippets laid out
# and linked from here:
# https://github.com/2i2c-org/infrastructure/issues/1444#issuecomment-1187405324

FROM pangeo/pangeo-notebook:2022.07.13
# install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook jupyterlab \
    https://github.com/jupyterhub/jupyter-remote-desktop-proxy/archive/main.zip \
        # jupyter-remote-desktop-proxy enables us to visit the /desktop path
        # just like we visit the /lab path. Visiting /desktop provides us
        # with an actual remote desktop experience.
        #
        # NOTE: This package is not available on conda-forge, but available
        #       on PyPI as jupyter-desktop-server I think but maybe not.
        #
        # NOTE: This install requires websockify to be installed via
        #       conda-forge. We have also installed TurboVNC for performance
        #       I think, and also various apt packages to get a desktop UI.

# create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}

# Install TurboVNC (https://github.com/TurboVNC/turbovnc)
ARG TURBOVNC_VERSION=2.2.6
RUN wget -q "https://sourceforge.net/projects/turbovnc/files/${TURBOVNC_VERSION}/turbovnc_${TURBOVNC_VERSION}_amd64.deb/download" -O turbovnc.deb \
 && apt-get install -y ./turbovnc.deb > /dev/null \
    # remove light-locker to prevent screen lock
 && apt-get remove -y light-locker > /dev/null \
 && rm ./turbovnc.deb \
 && ln -s /opt/TurboVNC/bin/* /usr/local/bin/
