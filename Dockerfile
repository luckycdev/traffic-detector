FROM python:3.11-slim

# Defaults port to 5050
ARG SERVER_PORT=5050

# Use port from docker-compose if present (either 5050 or SERVER_PORT from .env)
ENV SERVER_PORT=${SERVER_PORT}

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Records which port the app listens on
EXPOSE ${SERVER_PORT}

# Run main.py on container startup
CMD ["python", "main.py"]