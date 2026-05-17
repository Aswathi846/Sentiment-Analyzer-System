# 1. Use an official lightweight Python runtime as a parent image
FROM python:3.10-slim

# 2. Set system environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies required for build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy just the requirements first to leverage Docker caching layers
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application source code
COPY . .

# 8. Expose the default Streamlit port
EXPOSE 8501

# 9. Healthcheck to ensure the container is running smoothly
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. Command to run the Streamlit dashboard
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
