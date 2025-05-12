FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# System setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-venv python3.8-dev python3-pip git curl jq sudo \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Setup venv
RUN python -m venv /opt/mlops_env
ENV PATH="/opt/mlops_env/bin:$PATH"

# Install requirements
COPY ./training_scripts/arcface_torch/requirement.txt /tmp/requirement.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirement.txt

# Patch mxnet if needed
RUN MXNET_UTILS_FILE="/opt/mlops_env/lib/python3.8/site-packages/mxnet/numpy/utils.py" && \
    if [ -f "$MXNET_UTILS_FILE" ]; then \
      sed -i 's/bool_ = onp.bool_/bool_ = bool/' "$MXNET_UTILS_FILE" && \
      sed -i 's/bool = onp.bool/bool = bool/' "$MXNET_UTILS_FILE"; \
    fi

# Copy training code
COPY ./training_scripts /app/training_scripts
WORKDIR /app/training_scripts/arcface_torch

ENTRYPOINT ["bash", "run.sh"]
