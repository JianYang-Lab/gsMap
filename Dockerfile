FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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

## Install Python build tools
#RUN pip install --no-cache-dir --upgrade pip setuptools wheel flit-core

# Install the package and its dependencies using uv
RUN uv pip install --system --no-cache .

# Set the entrypoint to gsmap command
ENTRYPOINT ["gsmap"]

# Default command (can be overridden)
CMD ["--help"]