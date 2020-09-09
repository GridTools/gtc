FROM gitpod/workspace-full

USER root

RUN apt-get update -qq && \
    apt-get install -qq -y \
    libboost-all-dev \
    ccache && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run && \
    sudo sh cuda_11.0.3_450.51.06_linux.run --silent --toolkit && \
    rm cuda_11.0.3_450.51.06_linux.run && \
    echo "PATH=\$PATH:/usr/local/cuda-11.0/bin" >> $HOME/.bashrc

ARG ECBUILD_VERSION=3.3.2
RUN cd /tmp && \
    wget -q https://github.com/ecmwf/ecbuild/archive/${ECBUILD_VERSION}.tar.gz && \
    tar xzf ${ECBUILD_VERSION}.tar.gz && \
    mv ecbuild-${ECBUILD_VERSION} $HOME && \
    rm -rf /tmp/* && \
    echo "PATH=\$PATH:\$HOME/ecbuild-${ECBUILD_VERSION}/bin" >> $HOME/.bashrc

ARG ECKIT_VERSION=1.10.1
RUN export PATH=$PATH:$HOME/ecbuild-${ECBUILD_VERSION}/bin && \
    cd /tmp && \
    wget -q https://github.com/ecmwf/eckit/archive/${ECKIT_VERSION}.tar.gz && \
    tar xzf ${ECKIT_VERSION}.tar.gz && \
    cd eckit-${ECKIT_VERSION} && \
    mkdir build && cd build && \
    ecbuild .. && \
    make -j$(nproc) install && \
    rm -rf /tmp/*


ARG ATLAS_VERSION=0.20.1
RUN export PATH=$PATH:$HOME/ecbuild-${ECBUILD_VERSION}/bin && \
    cd /tmp && \
    wget -q https://github.com/ecmwf/atlas/archive/${ATLAS_VERSION}.tar.gz && \
    tar xzf ${ATLAS_VERSION}.tar.gz && \
    cd atlas-${ATLAS_VERSION} && \
    mkdir build && cd build && \
    ecbuild .. && \
    make -j$(nproc) install && \
    rm -rf /tmp/*

USER gitpod

COPY .gitpod/aliases.txt .
RUN cat aliases.txt >> $HOME/.bashrc

RUN pyenv install 3.8.2; pyenv global 3.8.2
