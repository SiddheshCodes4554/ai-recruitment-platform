# Use Python 3.11 slim image for a smaller footprint
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install system dependencies (build-essential for compiling any native deps if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir tf-keras --no-deps

# Copy source code and files
COPY src/ ./src/
COPY ranking_pipeline.py .
COPY rank.py .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Write a bootstrap script to run both services
RUN echo '#!/bin/bash\n\
uvicorn src.api:app --host 0.0.0.0 --port 8000 &\n\
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Run both backend and frontend services
ENTRYPOINT ["/app/entrypoint.sh"]
