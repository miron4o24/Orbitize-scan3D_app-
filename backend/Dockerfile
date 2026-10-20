# CPU-only photogrammetry backend: COLMAP (sparse SfM) + OpenMVS (dense) + Python mesh cleanup.
# Builds natively on amd64 and arm64 (e.g. Oracle Ampere) from the same file.
# Debian 13: OpenMVS 2.4 needs OpenCV >= 4.10 (Ubuntu 24.04 has 4.6).
FROM debian:trixie

ARG OPENMVS_TAG=v2.4.0
ARG BUILD_JOBS=2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git build-essential cmake pkg-config \
      libeigen3-dev libcgal-dev libopencv-dev libnanoflann-dev libjxl-dev \
      libboost-iostreams-dev libboost-program-options-dev libboost-system-dev libboost-serialization-dev \
      libpng-dev libjpeg-dev libtiff-dev libgmp-dev libmpfr-dev zlib1g-dev \
      colmap python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# IPO off: link-time optimisation needs several GB of RAM per compile job.
RUN git clone --depth 1 https://github.com/cdcseacave/VCG.git /tmp/vcglib \
 && git clone --depth 1 --branch ${OPENMVS_TAG} https://github.com/cdcseacave/openMVS.git /tmp/openMVS \
 # shortcut: OpenMVS 2.4 uses cv::IMWRITE_JPEGXL_QUALITY (OpenCV >= 4.11) only for .jxl output we never write;
 # use its numeric value (640) until Debian ships OpenCV 4.11+.
 && sed -i 's/cv::IMWRITE_JPEGXL_QUALITY/640/' /tmp/openMVS/libs/Common/Types.inl \
 && cmake -S /tmp/openMVS -B /tmp/openMVS_build -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT=/tmp/vcglib \
      -DOpenMVS_USE_CUDA=OFF -DOpenMVS_BUILD_VIEWER=OFF -DOpenMVS_USE_PYTHON=OFF \
      -DOpenMVS_ENABLE_TESTS=OFF -DOpenMVS_USE_BREAKPAD=OFF -DOpenMVS_ENABLE_IPO=OFF \
 && cmake --build /tmp/openMVS_build -j${BUILD_JOBS} \
 && cmake --install /tmp/openMVS_build \
 && rm -rf /tmp/openMVS /tmp/openMVS_build /tmp/vcglib
ENV PATH=/usr/local/bin/OpenMVS:/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv && pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /srv
COPY app ./app
COPY tests ./tests
ENV DATA_DIR=/data QT_QPA_PLATFORM=offscreen
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
