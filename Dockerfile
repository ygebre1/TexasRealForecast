# Use the official Python image from Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary files first (this helps Docker cache dependencies properly)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app

# Expose the port that the Dash app will run on
EXPOSE 8080

# Set the command to run your Dash app
CMD ["python", "app.py"]
