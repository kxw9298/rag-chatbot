# Start from the base Jupyter notebook image
FROM jupyter/scipy-notebook:latest

# Switch to root to install system packages
USER root

# Install PostgreSQL development libraries (libpq-dev)
RUN apt-get update && apt-get install -y libpq-dev

# Switch back to the default user (jovyan)
USER jovyan

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
