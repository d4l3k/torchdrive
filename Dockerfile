FROM nvcr.io/nvidia/pytorch:23.03-py3

# Install Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda update -n base -c defaults conda

# Create a conda virtual environment with Python 3.8
RUN conda create -n py38 python=3.8
ENV CONDA_DEFAULT_ENV="py38"
ENV CONDA_PREFIX="/root/miniconda3/envs/$CONDA_DEFAULT_ENV"
ENV PATH="$CONDA_PREFIX/bin:${PATH}"

# Link to pre-installed torch and torchvision
RUN ln -s /usr/local/lib/python3.8/dist-packages/torch $CONDA_PREFIX/lib/python3.8/site-packages/torch \
    && ln -s /usr/local/lib/python3.8/dist-packages/torchvision $CONDA_PREFIX/lib/python3.8/site-packages/torchvision

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install ffmpeg libsm6 libxext6 -y

# Install nuscenes
RUN pip install \
    nuscenes-devkit \
    black \
    pytest \
    pyre-check \
    scipy \
    matplotlib \
    orjson \
    tensorboard \
    GitPython \
    IPython \
    parameterized \
    av \
    torchinfo \
    sympy \
    torchx

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# From https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#building--installing-from-source
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip install openmim && mim install mmcv mmengine mmdet
