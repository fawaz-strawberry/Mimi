# Use the official Python 3.10 image as the base image
FROM python:3.10-slim-buster

# Optionally, install any additional dependencies
# RUN pip install some-package

# Set the working directory
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .
