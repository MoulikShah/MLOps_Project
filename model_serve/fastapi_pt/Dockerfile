# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install the OpenCV GUI dependency
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure model file is available inside the container
COPY model.pth /app/

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
