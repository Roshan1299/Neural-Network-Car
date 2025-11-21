# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies for pygame and other libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-mixer-dev \
        libsdl2-ttf-dev \
        libjpeg-dev \
        libpng-dev \
        libfreetype6-dev \
        libportmidi-dev \
        libx11-6 \
        libxv1 \
        libglu1-mesa-dev \
        libglib2.0-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ffmpeg \
        xvfb \
        x11-utils \
        x11-xserver-utils \
        xserver-xorg-video-all \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Copy the rest of the application code
COPY . /app/

# Set up for headless operation (for CI/CD or when no display is available)
ENV SDL_VIDEODRIVER=fbcon

# Expose a port (useful if we need to serve results via a web interface)
EXPOSE 8000

# Default command - will be overridden by docker-compose
CMD ["python", "main.py"]