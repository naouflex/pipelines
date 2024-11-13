FROM python:3.11-slim-bookworm AS base

# Use args
ARG MINIMUM_BUILD
ARG USE_CUDA
ARG USE_CUDA_VER

## Basis ##
ENV ENV=prod \
    PORT=9099 \
    # pass build args to the build
    MINIMUM_BUILD=${MINIMUM_BUILD} \
    USE_CUDA_DOCKER=${USE_CUDA} \
    USE_CUDA_DOCKER_VER=${USE_CUDA_VER}

# Install GCC, build tools, and Playwright dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    curl \
    git \
    # Playwright dependencies
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    # Additional dependencies that might be needed
    libx11-6 \
    libxext6 \
    libwayland-client0 \
    libwayland-cursor0 \
    libwayland-egl1 \
    libwayland-server0 \
    xvfb \
    libxshmfence1 \
    libglu1-mesa \
    libegl1 \
    libxdamage1 \
    libxfixes3 \
    libasound2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY ./requirements.txt .
COPY ./requirements-minimum.txt .
RUN pip3 install uv
RUN if [ "$MINIMUM_BUILD" != "true" ]; then \
        if [ "$USE_CUDA_DOCKER" = "true" ]; then \
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$USE_CUDA_DOCKER_VER --no-cache-dir; \
        else \
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir; \    
        fi \
    fi
RUN if [ "$MINIMUM_BUILD" = "true" ]; then \
        uv pip install --system -r requirements-minimum.txt --no-cache-dir; \
    else \
        uv pip install --system -r requirements.txt --no-cache-dir; \
    fi

# Install Playwright and browsers with system dependencies
RUN pip install --no-cache-dir playwright && \
    playwright install --with-deps chromium && \
    playwright install-deps chromium

# Copy the application code
COPY . .

# Expose the port
ENV HOST="0.0.0.0"
ENV PORT="9099"

ENTRYPOINT [ "bash", "start.sh" ]
