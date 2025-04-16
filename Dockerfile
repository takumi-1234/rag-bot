# Dockerfile (修正版 - 非rootユーザー対応)
FROM python:3.10-slim

# Create a non-root user and group
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID -o appgroup && \
    useradd --uid $USER_ID --gid $GROUP_ID --create-home --shell /bin/bash appuser

# Install build tools, supervisor and clean apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

WORKDIR /app

# Install dependencies (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Copy application code after installing deps and before changing permissions
COPY ./src /app/src

# Create directories expected by volumes and cache, and set permissions
# Include Hugging Face cache directory within /app
RUN mkdir -p /app/data/chroma /app/static/uploads /app/.cache/huggingface && \
    chown -R appuser:appgroup /app && \
    # Ensure the new user can write to supervisor log pipes (if needed, often managed by supervisor itself)
    # Ensure the new user can write to necessary directories
    chmod -R 755 /app # Give user execute permissions on directories

# Set the default user for the container
USER appuser

# Expose ports (Informational)
EXPOSE 8000
EXPOSE 8501

# Set Hugging Face cache directory via environment variable
ENV HF_HOME=/app/.cache/huggingface
# Optional: Set Streamlit home if needed (usually defaults to user's home)
# ENV STREAMLIT_HOME=/app/.streamlit

# Run supervisor to manage processes (will run as appuser by default now, but specify in conf too)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
