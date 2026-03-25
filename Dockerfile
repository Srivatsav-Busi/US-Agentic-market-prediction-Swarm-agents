# Build the React frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Build the Python backend
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies if required by standard data science packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies directly to leverage docker caching
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir build \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir gunicorn

COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Copy the built React assets from the frontend stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Set environment variables for the runtime
ENV PYTHONPATH=/app/src
ENV APP_HOST=0.0.0.0
ENV PORT=8000
ENV RESULTS_DIR=/app/results
ENV FRONTEND_DIR=/app/frontend/dist

# Expose the API and UI port
EXPOSE 8000

# Run the concurrent scheduler and web server
ENTRYPOINT ["/app/docker-entrypoint.sh"]
