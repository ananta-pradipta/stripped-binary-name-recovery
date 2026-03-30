# ============================================================
# Function Name Recovery Web App
# Multi-stage Docker build with BAP + PyTorch + Streamlit
# ============================================================

FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    wget curl git file binutils \
    opam m4 pkg-config libgmp-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install BAP via opam
RUN opam init --disable-sandboxing --auto-setup -y && \
    eval $(opam env) && \
    opam install bap -y && \
    echo 'eval $(opam env)' >> /etc/bash.bashrc

# Set up opam env for all commands
ENV PATH="/root/.opam/default/bin:${PATH}"
ENV OPAM_SWITCH_PREFIX="/root/.opam/default"
ENV CAML_LD_LIBRARY_PATH="/root/.opam/default/lib/stublibs"
ENV OCAML_TOPLEVEL_PATH="/root/.opam/default/lib/toplevel"

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch Geometric dependencies (CPU-only for smaller image)
RUN pip3 install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html 2>/dev/null || \
    pip3 install --no-cache-dir pyg-lib torch-scatter torch-sparse 2>/dev/null || true

# Copy application
WORKDIR /app
COPY . /app/

# Download checkpoints on build (optional, can also be mounted)
# RUN python3 download_checkpoints.py

# Streamlit config
RUN mkdir -p /root/.streamlit
RUN echo '\
[server]\n\
port = 7860\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[theme]\n\
primaryColor = "#D32F2F"\n\
backgroundColor = "#FFFFFF"\n\
secondaryBackgroundColor = "#F5F5F5"\n\
textColor = "#1a1a2e"\n\
' > /root/.streamlit/config.toml

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
