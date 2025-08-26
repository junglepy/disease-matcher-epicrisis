# LIRICAL Service Production Docker Image
FROM openjdk:17-slim

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Environment variables
ENV LIRICAL_JAR=/app/lirical-cli-2.2.0.jar
ENV LIRICAL_DATA=/app/data
ENV PYTHONUNBUFFERED=1
ENV PORT=8083
ENV JAVA_HOME=/usr/local/openjdk-17

# Expose port
EXPOSE 8083

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8083/health || exit 1

# Run the application
CMD ["python3", "app.py"]