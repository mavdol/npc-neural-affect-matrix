FROM rust:latest

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    clang \
    lld \
    llvm \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/clang /usr/bin/clang-cl && \
    ln -sf /usr/bin/lld /usr/bin/lld-link && \
    ln -sf /usr/bin/llvm-ar /usr/bin/llvm-lib

RUN rustup target add x86_64-pc-windows-msvc

RUN cargo install xwin --locked

WORKDIR /app

RUN xwin --accept-license splat --output /opt/xwin

ENV CC_x86_64_pc_windows_msvc="clang-cl"
ENV CXX_x86_64_pc_windows_msvc="clang-cl"
ENV AR_x86_64_pc_windows_msvc="llvm-lib"
ENV CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER="lld-link"
ENV RUSTFLAGS="-Lnative=/opt/xwin/crt/lib/x86_64 -Lnative=/opt/xwin/sdk/lib/um/x86_64 -Lnative=/opt/xwin/sdk/lib/ucrt/x86_64"

ENV CFLAGS_x86_64_pc_windows_msvc="-target x86_64-pc-windows-msvc -I/opt/xwin/crt/include -I/opt/xwin/sdk/include/ucrt -I/opt/xwin/sdk/include/um -I/opt/xwin/sdk/include/shared"
ENV CXXFLAGS_x86_64_pc_windows_msvc="-target x86_64-pc-windows-msvc -I/opt/xwin/crt/include -I/opt/xwin/sdk/include/ucrt -I/opt/xwin/sdk/include/um -I/opt/xwin/sdk/include/shared -EHsc"

RUN wget -O onnxruntime-win.zip \
    "https://github.com/microsoft/onnxruntime/releases/download/v1.22.1/onnxruntime-win-x64-1.22.1.zip" \
    && unzip onnxruntime-win.zip \
    && mkdir -p /app/onnxruntime-windows \
    && cp -r onnxruntime-win-x64-1.22.1/* /app/onnxruntime-windows/ \
    && rm -rf onnxruntime-win.zip onnxruntime-win-x64-1.22.1/

ENV ORT_LIB_LOCATION=/app/onnxruntime-windows/lib
ENV ORT_SKIP_DOWNLOAD=true
ENV ORT_PREFER_DYNAMIC_LINK=true
ENV ORT_STRATEGY=system
ENV ORT_DYLIB_PATH=/app/onnxruntime-windows/lib
ENV SKIP_SETUP=1

RUN which clang-cl && which lld-link && which llvm-lib

CMD ["sh", "-c", "export RUSTFLAGS=\"$RUSTFLAGS -L /app/onnxruntime-windows/lib\" && cargo build --target x86_64-pc-windows-msvc --release --lib && mkdir -p /app/target/x86_64-pc-windows-msvc/release && find /app/onnxruntime-windows -name '*.dll' -exec cp -v {} /app/target/x86_64-pc-windows-msvc/release/ \\;"]
