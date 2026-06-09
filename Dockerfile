# =========================================================================
# 1. USE LIGHTWEIGHT BASE IMAGE & SET UP SECURITY ENVIRONMENT
# =========================================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and force unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a secure, unprivileged user matching Hugging Face constraints
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set up the secure working directory inside the user's home space
WORKDIR /app

# =========================================================================
# 2. RUNTIME DEPENDENCY COMPILATION
# =========================================================================
# Copy dependency manifest matching the explicit user ownership permissions
COPY --chown=user requirements.txt /app/requirements.txt

# Install packages safely into the local user space profile
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --user -r requirements.txt

# =========================================================================
# 3. CODE ARTIFACT DEPLOYMENT
# =========================================================================
# Copy the source engine arrays and optimized model binaries cleanly
COPY --chown=user ./src /app/src
COPY --chown=user ./model /app/model

# Expose the mandatory Hugging Face web application service port
EXPOSE 7860

# =========================================================================
# 4. EXECUTE MICROSERVICE ROUTER via UVICORN ENGINE
# =========================================================================
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]