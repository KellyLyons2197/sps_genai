# Use an official Python runtime as a parent image
   FROM python:3.13-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download and run the uv installer, then remove it
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin/:$PATH"
   # Use the image's Python 3.13 instead of downloading a newer one
   ENV UV_PYTHON_DOWNLOADS=never

# Set the working directory
WORKDIR /code

# Copy the dependency files
COPY pyproject.toml uv.lock /code/

# Install dependencies (including spaCy and its model)
RUN uv sync --frozen --no-install-project

# Copy the application code
COPY ./app /code/app

# Run the application
CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--port", "80"]