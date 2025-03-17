FROM tensorflow/tensorflow:2.3.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire infercode directory into the container
COPY src /app/infercode/
COPY dataset/tree-sitter-parsers-Linux /root/.tree-sitter/tree-sitter-parsers-Linux
# Install infercode dependencies without installing infercode itself
RUN pip3 install --no-cache-dir bidict==0.21.2 coloredlogs==15.0.1 dpu-utils==0.2.19 \
    keras-radam==0.15.0 numpy==1.19.4 protobuf==3.14.0 scikit-learn==0.24.0 \
    scipy==1.5.2 tqdm==4.55.1 sentencepiece==0.1.95 tree-sitter-parsers>=0.0.3

# Install infercode from the local directory
RUN cd /app/infercode/ && pip3 install -e .

# Set environment variables
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_CPP_MIN_LOG_LEVEL="3"

# Keep container running
CMD ["tail", "-f", "/dev/null"]