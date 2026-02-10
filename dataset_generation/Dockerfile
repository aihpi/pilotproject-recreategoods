FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && apt-get install -y wget bzip2 && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy


ENV PATH="/opt/conda/bin:$PATH"

COPY . /opt/dataset_generation

WORKDIR /opt/dataset_generation

RUN conda env create -f environment.yml

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate generate-dataset"

ENV PATH="/opt/conda/envs/generate-dataset/bin:$PATH"
ENV CONDA_DEFAULT_ENV=generate-dataset