# Docker Compose Setup Guide

This guide explains how to run the entire avatar service using docker-compose, including the API server, Temporal worker, Redis, and Temporal server.

## Prerequisites

1. **Docker and Docker Compose** installed on your host machine
2. **ECON repository** cloned and configured (see [ECON_SETUP.md](./ECON_SETUP.md))
3. **Docker socket access** - The containers need access to the host's Docker daemon to run ECON

## Architecture

When running via docker-compose, the architecture looks like this:

```
┌─────────────────────────────────────────────────────────┐
│                    Host Machine                          │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  avatar-api  │  │ avatar-worker │  │    redis     │ │
│  │  (FastAPI)   │  │  (Temporal)   │  │              │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            │                             │
│                    ┌───────▼────────┐                    │
│                    │   temporal     │                    │
│                    │    server      │                    │
│                    └────────────────┘                    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Docker Socket (bind mount)               │   │
│  │  Allows containers to run docker-compose         │   │
│  └──────────────────────────────────────────────────┘   │
│                            │                             │
│                    ┌───────▼────────┐                    │
│                    │  ECON Service  │                    │
│                    │ (via docker-   │                    │
│                    │  compose)      │                    │
│                    └────────────────┘                    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │      Shared Volume (avatar_workdir)               │   │
│  │  For file exchange between services               │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Setup Steps

### 1. Configure Environment Variables

Set the following environment variables:

```bash
# Path to ECON repository
export ECON_REPO_PATH=/path/to/ECON

# Host path for workdir (where files are exchanged)
# This should be an absolute path on the host
export AVATAR_WORKDIR=/absolute/path/to/workdir

# Optional: If workdir path differs between container and host
export AVATAR_WORKDIR_HOST=/absolute/path/to/workdir
```

Or edit `docker-compose.yaml` and update the volume mounts:

```yaml
volumes:
  - ${ECON_REPO_PATH:-../ECON}:/econ:ro
  - ${AVATAR_WORKDIR:-./workdir}:/app/workdir
```

**Important:** 
- The `AVATAR_WORKDIR` should be an **absolute path** on the host for best results
- The default `./workdir` is relative to where you run `docker-compose`
- This directory will be created automatically if it doesn't exist
- Both the avatar service and ECON will access files in this directory

### 2. Update ECON docker-compose.yaml

Your ECON `docker-compose.yaml` should use environment variables for volume mounts. The avatar service will set these when calling ECON:

```yaml
version: '3.8'

services:
  econ:
    build: .
    volumes:
      - ${INPUT_DIR}:/app/input:ro
      - ${OUTPUT_DIR}:/app/output
      - ./configs:/app/configs:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Important:** The paths in ECON's docker-compose.yaml (`/app/input`, `/app/output`) are **container paths**. The avatar service will mount host directories to these paths.

### 3. Update config.yaml

Update your `config.yaml` to use service names instead of `localhost`:

```yaml
TEMPORAL_PARAMETERS:
  SERVER_URL: temporal:7233  # Use service name, not localhost

REDIS_PARAMETERS:
  HOST: redis  # Use service name, not localhost
  PORT: 6379

ECON_PARAMETERS:
  COMPOSE_PATH: /econ/docker-compose.yaml  # Path inside container

STORAGE_PARAMETERS:
  OUTPUT_DIR: /app/workdir  # Shared volume path
```

### 4. How File Sharing Works

When running in docker-compose, file sharing between the avatar service and ECON works as follows:

1. **Avatar service creates files** in `/app/workdir` (bind-mounted from host `AVATAR_WORKDIR`)
2. **Avatar service detects it's in a container** and maps container paths to host paths
3. **Avatar service runs docker-compose** (on host via bind-mounted socket) with host paths:
   ```python
   env["INPUT_DIR"] = "/host/path/to/workdir/input"  # Host path
   env["OUTPUT_DIR"] = "/host/path/to/workdir"       # Host path
   ```
4. **ECON's docker-compose** receives these host paths and mounts them:
   ```yaml
   volumes:
     - ${INPUT_DIR}:/app/input:ro  # Mounts host /workdir/input → container /app/input
     - ${OUTPUT_DIR}:/app/output   # Mounts host /workdir → container /app/output
   ```
5. **Both services access the same host directory** - no file copying needed!

**Key Point:** Since docker-compose runs on the host (via bind-mounted socket), it needs host paths, not container paths. The avatar service automatically maps container paths to host paths when running in Docker.

### 5. Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f avatar-api
docker-compose logs -f avatar-worker
```

### 6. Access Services

- **API Server**: http://localhost:8000
- **Temporal UI**: http://localhost:8088
- **Redis**: localhost:6379

## Key Differences from Host Setup

| Aspect | Host Setup | Docker Compose Setup |
|--------|-----------|---------------------|
| **File Paths** | Host filesystem (e.g., `/tmp/avatar_output`) | Shared Docker volume (`/app/workdir`) |
| **Docker Access** | Direct access to docker-compose | Bind mount docker socket |
| **Service Discovery** | `localhost` | Service names (e.g., `redis`, `temporal`) |
| **ECON Path** | Absolute host path | Mounted volume path (`/econ`) |
| **Network** | Local network | Docker bridge network |

## Troubleshooting

### Issue: "Cannot connect to Docker daemon"

**Solution:** Ensure the docker socket is accessible:
```bash
# Check socket permissions
ls -la /var/run/docker.sock

# If needed, add your user to docker group
sudo usermod -aG docker $USER
```

### Issue: ECON can't find files

**Solution:** Check that:
1. The `avatar_workdir` volume is properly mounted
2. Paths in `econ_activity.py` use container paths (`/app/workdir`)
3. ECON's docker-compose.yaml uses environment variables for mounts

### Issue: Services can't connect to each other

**Solution:** 
1. Ensure services are on the same docker-compose network (they are by default)
2. Use service names, not `localhost` in config
3. Check `depends_on` conditions in docker-compose.yaml

### Issue: GPU not available in ECON

**Solution:** Ensure your host has GPU support and docker-compose has GPU configuration:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Development vs Production

### Development (Current Setup)
- All services in one docker-compose file
- Temporal server included
- Redis included
- Good for local development

### Production Recommendations
- Use managed Temporal Cloud or separate Temporal cluster
- Use managed Redis (AWS ElastiCache, etc.)
- Separate docker-compose files for different environments
- Use secrets management for sensitive config
- Consider Kubernetes for orchestration

## Environment Variables

You can override configuration via environment variables:

```bash
# Set ECON repository path
export ECON_REPO_PATH=/path/to/ECON

# Override config file
export AVATAR_SERVICE_CONFIG=/path/to/config.yaml

# Override output directory
export AVATAR_OUTPUT_DIR=/custom/path
```

## Cleanup

```bash
# Stop services
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Clean up workdir (if using default ./workdir)
rm -rf ./workdir
```

## Quick Start Example

```bash
# 1. Set environment variables
export ECON_REPO_PATH=/path/to/ECON
export AVATAR_WORKDIR=$(pwd)/workdir  # Use absolute path

# 2. Create workdir
mkdir -p workdir

# 3. Start services
docker-compose up -d

# 4. Check logs
docker-compose logs -f avatar-api
docker-compose logs -f avatar-worker

# 5. Test API
curl -X POST http://localhost:8000/create -H "Content-Type: application/json" -d '{"img": "..."}'
```

## Notes

- The docker socket bind mount gives containers access to the host's Docker daemon
- This is necessary for the avatar service to run `docker-compose` commands for ECON
- The shared volume (`avatar_workdir`) allows file exchange without copying
- All services communicate via Docker's internal network
- GPU access requires nvidia-docker or Docker with GPU support
