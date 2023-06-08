# Base image
FROM python:3.8-slim-buster
# EXPOSE 8501

# Set the working directory in the container
WORKDIR /sales_app

# Copy the Python scripts and other files to the working directory
COPY requirements.txt .
COPY artifacts /sales_app/artifacts
COPY read_datasets.py .
COPY datasets /sales_app/datasets
COPY BuildTsModel.py .
COPY app.py .

# Install the packages with specific versions from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]

