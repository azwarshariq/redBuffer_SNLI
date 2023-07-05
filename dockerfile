# Use the official Python base image
FROM python:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the source code to the container
COPY src /app/src

# Copy the requirements file to the container
COPY requirements.txt /app

# Copy the data folder to the container
COPY snli_1.0 /app/snli_1.0

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the Python script
CMD [ "python", "src/main.py" ]
