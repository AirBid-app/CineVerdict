# Use an official, lightweight Python 3.11 slim image
FROM python:3.11-slim

# Install minimal build tools for compiling python packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the complete CineVerdict application code
COPY . .

# Expose standard port for Google Cloud Run
EXPOSE 8080

# Command to launch the ADK Web server.
# Uses 0.0.0.0 binding and disables persistent local storage for stateless container environments.
CMD ["adk", "web", "--host", "0.0.0.0", "--port", "8080", "--no-reload", "--no_use_local_storage", "cineverdict_agent"]
