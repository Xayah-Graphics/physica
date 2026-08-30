# syntax=docker/dockerfile:1
# hadolint global ignore=DL3007

FROM archlinux:latest AS build

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/pacman/pkg,sharing=locked \
    pacman -Syu --noconfirm --needed \
        base-devel \
        cmake \
        cuda \
        ffmpeg \
        git \
        ninja

ENV PATH="/opt/cuda/bin:${PATH}"

WORKDIR /src
COPY --link . .

RUN cmake -S . -B cmake-build-release -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
        -DPHYSICA_EXAMPLES=ON \
        -DPHYSICA_BUILD_SPECTRA=OFF \
        -DBUILD_TESTING=ON \
    && cmake --build cmake-build-release --parallel 30


FROM archlinux:latest AS runtime

RUN --mount=type=cache,target=/var/cache/pacman/pkg,sharing=locked \
    pacman -Syu --noconfirm --needed cuda gcc-libs \
    && groupadd --gid 10001 physica \
    && useradd --uid 10001 --gid 10001 --home-dir /workspace --shell /usr/bin/nologin physica \
    && install --directory --owner=10001 --group=10001 /opt/physica/bin /workspace

COPY --from=build --chown=10001:10001 --link /src/cmake-build-release/physica-example-instant-ngp-cli /opt/physica/bin/instant-ngp

USER 10001:10001
ENV HOME=/workspace
WORKDIR /workspace

ENTRYPOINT ["/opt/physica/bin/instant-ngp"]
