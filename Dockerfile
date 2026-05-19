# Build image
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS build

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    libssl-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.11.0%2Bcu126.zip && \
    unzip libtorch-shared-with-deps-2.11.0+cu126.zip && \
    cd libtorch && \
    cp -r include/* /usr/include && \
    cp -r lib/* /usr/lib && \
    cp -r share/* /usr/share

ENV PATH="/usr/local/cargo/bin:${PATH}"
ENV CARGO_HOME="/usr/local/cargo"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path

COPY . /app
WORKDIR /app
RUN cargo build --release --features cuda

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run image
# Doesn't worky with runtime image, it needs nvcc when training
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS run

COPY --from=build /tmp/libtorch/include/* /usr/include
COPY --from=build /tmp/libtorch/lib/* /usr/lib
COPY --from=build /tmp/libtorch/share/* /usr/share
COPY --from=build /app/target/release/burn-playground /usr/local/bin/burn-playground

ENTRYPOINT ["/usr/local/bin/burn-playground"]
