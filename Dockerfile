# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase into the container
COPY . .

# Set environment variables
ENV SHELDOR_DEFAULT_LLM_MODEL=Meta-Llama-3-8B-Instruct
ENV SHELDOR_LOG_LEVEL=INFO

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"] 