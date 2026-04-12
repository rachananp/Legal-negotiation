FROM ghcr.io/meta-pytorch/openenv-base:latest

# Enable the web UI at http://localhost:8000/web
ENV ENABLE_WEB_INTERFACE=true

WORKDIR /app

# Install dependencies first (cached layer)
RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir fastapi uvicorn

# Install latest openenv from GitHub
RUN pip install --no-cache-dir git+https://github.com/meta-pytorch/openenv.git

# Copy ALL project files into the container
COPY . /app
RUN pip install --no-cache-dir -e .
# Run the server using Python directly (avoids any uv/module issues)
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
