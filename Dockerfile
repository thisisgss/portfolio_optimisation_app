# Use a Python 3.11 slim image for a smaller footprint
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (needed for compilation of some packages and fonts for FPDF/Kaleido)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    fontconfig \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
# Note: fpdf2 and kaleido are included in your requirements.txt 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
# This copies app.py and any other necessary files
COPY . .

# Environment variable to control Streamlit data collection
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck to verify the Streamlit server is responsive
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the Streamlit application
# We use the correct file name (app.py) and ensure it binds to 0.0.0.0
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
