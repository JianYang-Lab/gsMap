FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libhdf5-dev \
    libgsl-dev \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . /app/

# CPU image: PyTorch from the CPU wheel index and CPU JAX, so both frameworks
# share one runtime (a plain install would pull CUDA PyTorch next to CPU JAX).
RUN uv pip install --system --no-cache --torch-backend=cpu ".[cpu]"

# Set the entrypoint to gsmap command
ENTRYPOINT ["gsmap"]

# Default command (can be overridden)
CMD ["--help"]
