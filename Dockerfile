# =========================================================================
# 1. USE AN OFFICIAL LIGHTWEIGHT PYTHON BASE IMAGE
# =========================================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and force unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =========================================================================
# 2. SET UP THE WORKING DIRECTORY INSIDE THE CONTAINER
# =========================================================================
WORKDIR /app

# Install system utilities needed for compiling certain dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================================================================
# 3. CACHE DEPENDENCIES LAYERS FOR FASTER BUILD SPEEDS
# =========================================================================
COPY requirements.txt /app/

# Install dependencies directly into the system container (no virtualenv needed inside Docker)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# =========================================================================
# 4. COPY COMPREHENSIVE PROJECT ARTIFACTS
# =========================================================================
COPY ./src /app/src
COPY ./model /app/model

# 🌟 FIX: Expose port 7860 for Hugging Face Spaces compatibility
EXPOSE 7860

# =========================================================================
# 5. EXECUTE PRODUCTION API SERVICE VIA UVICORN
# =========================================================================
# 🌟 FIX: Map the Uvicorn engine directly to serve traffic over port 7860
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]