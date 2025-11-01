# Use the standard Python 3.10 runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first for layer caching
COPY requirements.txt .

# Update package lists and install necessary system dependencies for OpenCV
# Includes libraries that often help headless OpenCV function correctly
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgtk2.0-dev && \
    # Clean up apt lists to reduce image size
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (app.py, .keras model, .xml cascade)
# into the container at /app
COPY . .

# Make port 7860 available to the world outside this container
# Hugging Face Spaces typically expects apps on this port
EXPOSE 7860

# Define environment variable (optional, good practice)
ENV NAME FacialExpressionAPI

# Run app.py when the container launches using Gunicorn
# It will listen on all interfaces (0.0.0.0) on port 7860
# 'app:app' means: find the file 'app.py' and run the Flask object named 'app'
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]

