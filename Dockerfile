# Dockerfile for cross-compiling to Windows
FROM rust:latest

# Install Windows cross-compilation tools and utilities
RUN apt-get update && apt-get install -y \
    mingw-w64 \
    gcc-mingw-w64-x86-64 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Add Windows target
RUN rustup target add x86_64-pc-windows-gnu

# Set environment variables for cross-compilation
ENV CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc
ENV CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc
ENV CXX_x86_64_pc_windows_gnu=x86_64-w64-mingw32-g++

# Set working directory
WORKDIR /app

# Download and setup ONNX Runtime Windows binaries
RUN wget -O onnxruntime-win.zip \
    "https://github.com/microsoft/onnxruntime/releases/download/v1.22.1/onnxruntime-win-x64-1.22.1.zip" \
    && unzip onnxruntime-win.zip \
    && ls -la onnxruntime-win-x64-1.22.1/ \
    && mkdir -p /app/onnxruntime-windows \
    && cp -r onnxruntime-win-x64-1.22.1/* /app/onnxruntime-windows/ \
    && ls -la /app/onnxruntime-windows/ \
    && find /app/onnxruntime-windows -name "*.dll" -type f \
    && rm -rf onnxruntime-win.zip onnxruntime-win-x64-1.22.1/

# Set ONNX Runtime environment variables for cross-compilation
ENV ORT_LIB_LOCATION=/app/onnxruntime-windows/lib
ENV ORT_SKIP_DOWNLOAD=true
ENV ORT_PREFER_DYNAMIC_LINK=true
ENV ORT_STRATEGY=system
ENV RUSTFLAGS="-L /app/onnxruntime-windows/lib"
ENV ORT_DYLIB_PATH=/app/onnxruntime-windows/lib

# Copy source code
COPY . .

# Skip model setup during cross-compilation
ENV SKIP_SETUP=1

# Build the project with proper DLL copying
CMD ["sh", "-c", "echo '🔍 Checking ONNX Runtime directory structure:' && \
    ls -la /app/onnxruntime-windows/ && \
    ls -la /app/onnxruntime-windows/lib/ && \
    find /app/onnxruntime-windows -name '*.dll' -type f && \
    find /app/onnxruntime-windows -name '*.lib' -type f && \
    echo '🔨 Building project...' && \
    RUSTFLAGS='-L /app/onnxruntime-windows/lib' cargo build --target x86_64-pc-windows-gnu --release --lib && \
    echo '📋 Creating target directory and copying DLLs...' && \
    mkdir -p /app/target/x86_64-pc-windows-gnu/release && \
    find /app/onnxruntime-windows -name '*.dll' -exec cp -v {} /app/target/x86_64-pc-windows-gnu/release/ \\; && \
    echo '📦 Final build contents:' && \
    ls -la /app/target/x86_64-pc-windows-gnu/release/ && \
    echo '✅ Build complete! Windows binaries and DLLs are in target/x86_64-pc-windows-gnu/release/'"]
