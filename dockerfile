FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# 2) Install system tools, Python 3.10 & pip, venv support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 \
      python3.10 \
      python3.10-venv \          
      python3-pip \     
      git \
      wget \
      unzip && \
    rm -rf /var/lib/apt/lists/*

# 3) Install uv via Astral’s installer (with wget)
RUN wget -qO- https://astral.sh/uv/install.sh | sh


# 4) Clone your repo into /workspace
WORKDIR /workspace
RUN git clone -b saigum https://github.com/rbSparky/MGT-PlusPlus.git

# 5) Create the uv venv & install your Python deps
WORKDIR /workspace/MGT-PlusPlus
RUN python3 -m venv mgt 
ENV PATH="/workspace/MGT-PlusPlus/mgt/bin:${PATH}"

# Copy and install your Ada-style requirements
COPY ada.txt ./
RUN pip install --no-cache-dir -r ada.txt \
    && pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html \
    && pip install wandb


# 6) Download & preprocess QMOF dataset
RUN wget https://figshare.com/ndownloader/files/51716795 -O qmof.zip \
    && unzip qmof.zip \
    && cd qmof_database/qmof_database \
    && unzip relaxed_structures.zip \
    && cd /workspace/MGT-PlusPlus \
    && python script.py \
         --qmof_json qmof_database/qmof_database/qmof.json \
         --structures_dir qmof_database/qmof_database/relaxed_structures \
         --output_dir qmof_database/qmof_database/ \
    && cp examples/example_data/atom_init.json qmof_database/qmof_database/atom_init.json \
    && mv qmof_database/qmof_database/relaxed_structures qmof_database/qmof_database/raw

# 7) Default workdir & command
WORKDIR /workspace/MGT-PlusPlus
RUN pip install torch_geometric
CMD ["bash", "-c", "while true; do sleep 3600; done"]
