FROM python:3.11-slim

ARG SERVER_PORT=5050
ENV SERVER_PORT=${SERVER_PORT}

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE ${SERVER_PORT}

CMD ["python", "main.py"]