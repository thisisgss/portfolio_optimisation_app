# Use an official Python runtime as a parent image
# We choose a lightweight, smaller image for better performance
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY app.py .

# Tell Docker that the container listens on the specified network port at runtime
# Streamlit's default port is 8501
EXPOSE 8501

# Set the environment variable to make Streamlit run on all interfaces
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Define the command to run the Streamlit application
# We use the full command to avoid issues with Python/Streamlit pathing
CMD ["streamlit", "run", "app.py"]