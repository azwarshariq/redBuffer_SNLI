# Use the official Python base image
FROM python:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt /app

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code to the container
COPY src /app/src

# Copy the data folder to the container
COPY snli_1.0 /app/snli_1.0

# Set the command to run the Python script
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "src.flask_app:app"]
