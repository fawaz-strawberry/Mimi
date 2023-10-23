#!/bin/bash

# Build the Docker image
docker build -t mimi .

# Run the Docker container
docker run mimi /bin/bash -c 'python script.py && python validate.py'
