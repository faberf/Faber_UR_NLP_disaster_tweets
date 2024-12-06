# syntax=docker/dockerfile:1

FROM mcr.microsoft.com/devcontainers/base:jammy

# Install Miniforge (Conda)
RUN curl -fsSL -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3.sh -b -p /opt/conda \
    && rm Miniforge3.sh

# Update PATH
ENV PATH=/opt/conda/bin:$PATH

# Initialize Conda
RUN conda init bash && conda init zsh

# Install Mamba in the base environment
RUN conda install -n base -c conda-forge mamba -y

# Copy environment.yml
COPY environment.yml /tmp/environment.yml

# Use Mamba to create the environment with cache mount
RUN --mount=type=cache,target=/opt/conda/pkgs mamba env create -f /tmp/environment.yml \
    && conda clean -a -y

# Set environment variables
ENV CONDA_DEFAULT_ENV=dev_env
ENV PATH=/opt/conda/envs/dev_env/bin:$PATH

# Activate the environment for subsequent RUN commands
SHELL ["conda", "run", "-n", "dev_env", "/bin/bash", "-c"]

# Download NLTK data with cache mount
RUN --mount=type=cache,target=/root/nltk_data python -m nltk.downloader all

# Set the default command
CMD ["bash"]
