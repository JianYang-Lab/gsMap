# Docker CI/CD Setup Documentation

## Overview

This project has automated Docker image building and publishing through GitHub Actions. When you create a new version tag, the CI/CD pipeline automatically:

1. Builds Python packages and publishes to PyPI
2. Builds multi-architecture Docker images (amd64, arm64)
3. Publishes Docker images to GitHub Container Registry (ghcr.io)
4. Creates a GitHub Release with all artifacts

## How to Trigger a Release

```bash
# Create and push a version tag
git tag v1.0.0
git push origin v1.0.0

# Or create an annotated tag with a message
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

## Docker Image Locations

After a successful release, Docker images are available at:

- **GitHub Container Registry (Primary)**: `ghcr.io/jianyang-lab/gsmap`
- **Docker Hub (Optional)**: `<your-dockerhub-username>/gsmap` (requires setup)

## Available Tags

For each release (e.g., v1.2.3), the following tags are created:

- `latest` - Always points to the most recent release
- `1.2.3` - Exact version
- `1.2` - Major.minor version (updates with patches)
- `1` - Major version (updates with minor/patches)
- `sha-<commit>` - Specific commit SHA

## Setting Up Docker Hub (Optional)

If you want to also push images to Docker Hub:

1. Create a Docker Hub account
2. Generate an access token at https://hub.docker.com/settings/security
3. Add these secrets to your GitHub repository:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token

Go to: Settings → Secrets and variables → Actions → New repository secret

## CI/CD Workflows

### 1. `docker-publish.yml`
- Triggers on version tags (v*)
- Builds and pushes to ghcr.io
- Optionally pushes to Docker Hub if secrets are configured
- Supports manual workflow dispatch

### 2. `release.yml`
- Complete release pipeline
- Combines PyPI and Docker publishing
- Creates GitHub Release with all artifacts
- Generates release notes automatically

### 3. `publish-to-pypi.yml`
- Original PyPI publishing workflow
- Still functional for Python-only releases

## Multi-Architecture Support

Images are built for both:
- `linux/amd64` (Intel/AMD processors)
- `linux/arm64` (Apple Silicon, ARM servers)

This ensures compatibility across different platforms.

## Permissions

The GitHub Actions workflows use the built-in `GITHUB_TOKEN` which automatically has permissions to:
- Push to GitHub Container Registry (ghcr.io)
- Create GitHub Releases
- Upload release artifacts

No additional configuration is needed for ghcr.io.

## Testing the Docker Build Locally

Before pushing a tag, you can test the Docker build:

```bash
# Build locally
docker build -t gsmap:test .

# Test the image
docker run --rm gsmap:test --help

# Test with data
docker run --rm -v $(pwd)/data:/data gsmap:test [command]
```

## Monitoring CI/CD

You can monitor the build status at:
- https://github.com/JianYang-Lab/gsMap/actions

## Troubleshooting

### Build Failures

1. Check the Actions tab for error logs
2. Common issues:
   - Missing dependencies in Dockerfile
   - Python package build errors
   - Network timeouts

### Image Not Found

If the image isn't available after release:
1. Check if the workflow completed successfully
2. Verify the tag format (must start with 'v')
3. Check permissions in repository settings

### Manual Trigger

You can manually trigger the Docker build:
1. Go to Actions → Docker Publish
2. Click "Run workflow"
3. Select branch and run

## Best Practices

1. **Test locally first**: Always build and test Docker images locally before creating a release tag
2. **Use semantic versioning**: Follow v1.2.3 format for consistent tagging
3. **Update dependencies**: Keep Dockerfile dependencies up to date
4. **Document changes**: Update CHANGELOG or release notes for each version
5. **Pin versions**: Consider pinning critical dependencies in Dockerfile for reproducibility