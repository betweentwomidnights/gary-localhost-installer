# Compile/test the Linux control center without installing desktop development
# packages on the host. Docker chooses ARM64 on Spark and x64 on an x64 host.
FROM rust:1.95-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/control-center/src-tauri
CMD ["cargo", "check", "--locked", "--all-targets"]
