#!/bin/bash

# Build script for Windows cross-compilation using Docker
echo "🚀 Building NPC Neural Affect Matrix for Windows..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/

# Build Docker image
echo "📦 Building Docker image..."
docker build -f Dockerfile -t npc-neural-affect-matrix:windows .

# Run the build
echo "🔨 Cross-compiling to Windows..."
# Create a temporary directory without the bin folder and temporary Cargo.toml
mkdir -p temp_src
cp -r src/* temp_src/ 2>/dev/null || true
rm -rf temp_src/bin 2>/dev/null || true

# Create temporary Cargo.toml without binary section
sed '/^\[\[bin\]\]/,/^$/d' Cargo.toml > temp_Cargo.toml

docker run --rm \
    -v "$(pwd)/temp_src:/app/src" \
    -v "$(pwd)/temp_Cargo.toml:/app/Cargo.toml" \
    -v "$(pwd)/Cargo.lock:/app/Cargo.lock" \
    -v "$(pwd)/build.rs:/app/build.rs" \
    -v "$(pwd)/target:/app/target" \
    npc-neural-affect-matrix:windows

# Clean up
rm -rf temp_src temp_Cargo.toml

# Verify the build output
echo "🔍 Verifying build output..."

# Check if target directory exists
if [ ! -d "target/x86_64-pc-windows-gnu/release" ]; then
    echo "❌ Target directory not found!"
    echo "📦 Available directories:"
    find target -type d -name "*windows*" 2>/dev/null || echo "No Windows target directories found"
    exit 1
fi

# Check for library output
if [ -f "target/x86_64-pc-windows-gnu/release/npc_neural_affect_matrix.dll" ]; then
    echo "✅ Library: npc_neural_affect_matrix.dll"
else
    echo "❌ Library not found!"
fi

# Check for ONNX Runtime DLLs
if [ -f "target/x86_64-pc-windows-gnu/release/onnxruntime.dll" ]; then
    echo "✅ ONNX Runtime DLL: onnxruntime.dll"
else
    echo "❌ onnxruntime.dll not found!"
fi

if [ -f "target/x86_64-pc-windows-gnu/release/onnxruntime_providers_shared.dll" ]; then
    echo "✅ ONNX Runtime Providers DLL: onnxruntime_providers_shared.dll"
else
    echo "❌ onnxruntime_providers_shared.dll not found!"
fi

echo "📦 Windows build contents:"
ls -la target/x86_64-pc-windows-gnu/release/ 2>/dev/null || echo "Directory listing failed"

# Create clean distribution folder with only essential files for Unity/Unreal
echo "📦 Creating clean distribution folder..."
rm -rf dist 2>/dev/null || true
mkdir -p dist

# Copy only the essential DLLs needed for Unity/Unreal Engine
if [ -f "target/x86_64-pc-windows-gnu/release/npc_neural_affect_matrix.dll" ]; then
    cp "target/x86_64-pc-windows-gnu/release/npc_neural_affect_matrix.dll" "dist/"
    echo "✅ Copied: npc_neural_affect_matrix.dll"
else
    echo "❌ npc_neural_affect_matrix.dll not found!"
fi

if [ -f "target/x86_64-pc-windows-gnu/release/onnxruntime.dll" ]; then
    cp "target/x86_64-pc-windows-gnu/release/onnxruntime.dll" "dist/"
    echo "✅ Copied: onnxruntime.dll"
else
    echo "❌ onnxruntime.dll not found!"
fi

if [ -f "target/x86_64-pc-windows-gnu/release/onnxruntime_providers_shared.dll" ]; then
    cp "target/x86_64-pc-windows-gnu/release/onnxruntime_providers_shared.dll" "dist/"
    echo "✅ Copied: onnxruntime_providers_shared.dll"
else
    echo "❌ onnxruntime_providers_shared.dll not found!"
fi

echo ""
echo "📦 Clean distribution contents:"
ls -la dist/ 2>/dev/null || echo "Distribution directory not found"

echo ""
echo "✅ Build complete!"
echo "📁 Full build artifacts: target/x86_64-pc-windows-gnu/release/"
echo "📁 Unity/Unreal ready DLLs: dist/"
echo ""
echo "🎮 For Unity/Unreal Engine or any other engine that supports .dll files, use only the files in dist/"
