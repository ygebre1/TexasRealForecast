# Use the official Python image from Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your app files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the Dash app will run on
EXPOSE 8080

# Set the command to run your Dash app
CMD ["python", "app.py"]
