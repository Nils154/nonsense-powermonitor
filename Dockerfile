FROM python:3.11-slim

# Set working directory
WORKDIR /srv/docker/powermonitor

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./
COPY templates/ ./templates/
COPY matplotlib_config.py ./

# Create data directory for database and plots
RUN mkdir -p /app/data /app/plots

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=nonsense_power_analyzer.py
ENV FLASK_ENV=production

# Expose port for web application
EXPOSE 8888

# Create startup script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Use entrypoint script to run both processes
ENTRYPOINT ["/docker-entrypoint.sh"]
