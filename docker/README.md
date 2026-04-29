# Docker Configuration

This directory contains the Docker configuration files required to run the local infrastructure components (PostgreSQL, Neo4j, Airflow, Qdrant).

## Key Files
- `requirements.txt`: Python dependencies needed for the Docker deployment.
- `airflow.Dockerfile` (if present): Custom Airflow image definition.

The core service definitions are managed in the `docker-compose.yml` file located in the repository root.
