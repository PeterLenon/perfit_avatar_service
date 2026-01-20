# Documentation

This folder contains technical documentation for the Perfit Avatar Service.

## Contents

| Document | Description |
|----------|-------------|
| [DESIGN.md](./DESIGN.md) | Architecture, design decisions, and system overview |
| [IMPLEMENTATION.md](./IMPLEMENTATION.md) | Code-level details, file structure, and how-to guides |

## Quick Links

### Design Document

- [System Architecture](./DESIGN.md#architecture)
- [Design Decisions](./DESIGN.md#design-decisions)
- [Data Models](./DESIGN.md#data-models)
- [API Design](./DESIGN.md#api-design)
- [ML Pipeline](./DESIGN.md#ml-pipeline)

### Implementation Guide

- [Project Structure](./IMPLEMENTATION.md#project-structure)
- [Configuration](./IMPLEMENTATION.md#1-configuration-appconfigpy)
- [API Routes](./IMPLEMENTATION.md#3-api-routes-appapiroutesavatarpy)
- [HMR2 Inference](./IMPLEMENTATION.md#6-hmr2-inference-mlhmr2inferencepy)
- [Measurement Extraction](./IMPLEMENTATION.md#7-measurement-extraction-mlmeasurementsextractorpy)
- [Docker Setup](./IMPLEMENTATION.md#docker-setup)
- [Running Locally](./IMPLEMENTATION.md#running-locally)
- [Common Issues](./IMPLEMENTATION.md#common-issues)

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/avatar` | Create avatar from photo |
| `GET` | `/api/v1/avatar/job/{job_id}` | Poll job status |
| `GET` | `/api/v1/avatar/{user_id}` | Get latest avatar |
| `GET` | `/api/v1/avatar/{user_id}/history` | Get all avatars |
| `GET` | `/api/v1/health` | Health check |

## Body Measurements Extracted

1. Height (cm)
2. Chest circumference (cm)
3. Waist circumference (cm)
4. Hip circumference (cm)
5. Inseam (cm)
6. Shoulder width (cm)
7. Arm length (cm)
8. Thigh circumference (cm)
9. Neck circumference (cm)
