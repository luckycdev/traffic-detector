FROM python:3.11-slim

# Defaults port to 5050
ARG SERVER_PORT=5050

# use port in .env if changed
ENV SERVER_PORT=${SERVER_PORT}

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Pull and install required tools and libraries from txt file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy files from local machine into docker container
COPY . .

# Tells docker to use specified port (default is 5050)
EXPOSE ${SERVER_PORT}

# Execute both python and main application file
CMD ["python", "main.py"]