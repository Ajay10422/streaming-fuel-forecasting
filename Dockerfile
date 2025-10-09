# Dockerfile
# This file defines the container image for all Python services.

# Use an official Python runtime as a parent image.
# Using a 'slim' version keeps the image size smaller.
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the Python dependencies. --no-cache-dir reduces image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .