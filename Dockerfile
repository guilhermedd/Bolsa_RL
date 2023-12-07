# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

# Create the models directory and change its ownership to the user running the app
# RUN mkdir -p models && chown -R $USER:$USER models

# Copy only the requirements file and install Python dependencies
COPY assets/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application files
COPY . .

# Define the command to run your application
CMD ["bash"]

CMD ["python", "src/actor_critic_method.py"]
