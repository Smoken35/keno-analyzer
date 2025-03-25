FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ scripts/
COPY prometheus/ prometheus/
COPY alertmanager/ alertmanager/

# Create log directory
RUN mkdir -p logs

# Expose ports
EXPOSE 8000 3456

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Use the service manager as entrypoint
ENTRYPOINT ["python", "scripts/manage_services.py"]
CMD ["start"] 