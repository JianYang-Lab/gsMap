# Docker Setup for gsMap

This guide explains how to build and run gsMap using Docker for reproducible analysis.

## Quick Start - Using Pre-built Images

The easiest way to use gsMap with Docker is to pull the pre-built image from GitHub Container Registry:

```bash
# Pull the latest image
docker pull ghcr.io/jianyang-lab/gsmap:latest

# Or pull a specific version
docker pull ghcr.io/jianyang-lab/gsmap:v1.0.0

# Run gsMap
docker run --rm ghcr.io/jianyang-lab/gsmap:latest --help
```

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

## Building the Docker Image

### Option 1: Using Docker directly

```bash
# Build the image
docker build -t gsmap:latest .
```

### Option 2: Using Docker Compose

```bash
# Build the image using docker-compose
docker-compose build
```

## Running gsMap in Docker

### Using Docker directly

```bash
# Show help
docker run --rm gsmap:latest --help

# Run with mounted data directory
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/results:/results \
  gsmap:latest [subcommand] [options]

# Example: Run a specific gsMap command
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/results:/results \
  gsmap:latest format-sumstats \
  --sumstats /data/input.txt \
  --out /results/output
```

### Using Docker Compose

```bash
# Show help
docker-compose run --rm gsmap --help

# Run a specific command
docker-compose run --rm gsmap [subcommand] [options]

# Example with custom command
docker-compose run --rm gsmap format-sumstats \
  --sumstats /data/input.txt \
  --out /results/output
```

## Volume Mounts

The Docker setup includes two default volume mounts:

- `./data:/data` - For input data files
- `./results:/results` - For output results

Place your input files in the `data/` directory and reference them with `/data/` prefix in commands.
Results will be saved to the `results/` directory.

## Interactive Shell

To get an interactive shell for debugging:

```bash
# Using Docker
docker run --rm -it \
  -v $(pwd)/data:/data \
  -v $(pwd)/results:/results \
  --entrypoint /bin/bash \
  gsmap:latest

# Using Docker Compose
docker-compose run --rm --entrypoint /bin/bash gsmap
```

## Tips for Reproducibility

1. **Pin versions**: Consider updating the Dockerfile to use specific versions of dependencies

1. **Document data**: Keep track of input data versions and sources

1. **Tag images**: Use version tags for your Docker images

    ```bash
    docker build -t gsmap:v1.0.0 .
    ```

1. **Save configurations**: Store your command parameters in scripts or configuration files

## Multi-platform builds (Optional)

For building images that work on different architectures (e.g., AMD64, ARM64):

```bash
# Setup buildx (one-time)
docker buildx create --name mybuilder --use

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t gsmap:latest --push .
```

## Troubleshooting

- If you encounter memory issues, increase Docker's memory limit in Docker Desktop settings
- For permission issues with mounted volumes, ensure the data and results directories have appropriate permissions
- If builds are slow, consider using a `.dockerignore` file to exclude unnecessary files
