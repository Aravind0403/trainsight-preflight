# Lightweight Python base image for fast K8s InitContainer startup
FROM python:3.11-slim

WORKDIR /app

# Prevent python from writing pyc files to disk and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install package dependencies
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

RUN pip install --no-cache-dir .

# Default entrypoint
ENTRYPOINT ["trainsight"]
CMD ["--help"]
