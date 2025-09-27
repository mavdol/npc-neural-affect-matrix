#!/bin/bash

echo "Building NPC Neural Affect Matrix binaries"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

rm -rf "${SCRIPT_DIR}/dist/"
rm -rf "${SCRIPT_DIR}/target/x86_64-pc-windows-msvc/"

docker build -f Dockerfile -t npc-neural-affect-matrix:msvc .

if [ $? -ne 0 ]; then
    echo "❌ Docker image build failed!"
    exit 1
fi


mkdir -p "${SCRIPT_DIR}/target/x86_64-pc-windows-msvc"

CONTAINER_NAME="npc-msvc-build-temp-$(date +%s)"

docker run --name "$CONTAINER_NAME" \
    -v "${SCRIPT_DIR}/src:/app/src" \
    -v "${SCRIPT_DIR}/Cargo.toml:/app/Cargo.toml" \
    -v "${SCRIPT_DIR}/Cargo.lock:/app/Cargo.lock" \
    -v "${SCRIPT_DIR}/build.rs:/app/build.rs" \
    -v "${SCRIPT_DIR}/target/x86_64-pc-windows-msvc/:/app/target/x86_64-pc-windows-msvc/" \
    npc-neural-affect-matrix:msvc

BUILD_RESULT=$?

if [ $BUILD_RESULT -ne 0 ]; then
    echo "❌ MSVC build failed!"
    docker rm "$CONTAINER_NAME" 2>/dev/null
    rm -rf "${SCRIPT_DIR}/target/x86_64-pc-windows-msvc/"
    exit 1
fi

docker cp "$CONTAINER_NAME:/app/target/x86_64-pc-windows-msvc/" "${SCRIPT_DIR}/target/"
docker rm "$CONTAINER_NAME"

mkdir -p dist
TARGET_DIR="target/x86_64-pc-windows-msvc/release"
MAIN_DLL=$(find "$TARGET_DIR" -name "*.dll" -type f ! -name "onnxruntime*" | head -1)

cp "$TARGET_DIR/onnxruntime.dll" "dist/"
cp "$MAIN_DLL" "dist/"
cp "$TARGET_DIR/onnxruntime_providers_shared.dll" "dist/"
