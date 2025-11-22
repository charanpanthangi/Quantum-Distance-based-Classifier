# Lightweight Python image to keep the container small
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /app

# Copy dependency list and install packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Default command runs the demo script
CMD ["python", "app/main.py"]
