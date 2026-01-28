# Data Ingestion Pipeline
This directory manages the "LLM-ETL" pipeline (The "Worker").

## Structure
- **/dags**: Airflow DAGs for scheduling tasks (e.g., daily ingestion, weekly retraining).
- **/etl**: Python scripts for extracting, transforming, and loading financial data.
- **/schema**: Database schema definitions for PostgreSQL and Qdrant.
