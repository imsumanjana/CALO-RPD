# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941
FROM ${PYTHON_IMAGE} AS runtime

ARG RUNTIME_LOCK=requirements-lock-cpu-py311-linux.txt
ARG SOURCE_COMMIT=unavailable
ARG SOURCE_TRACKED_CLEAN=false

LABEL org.opencontainers.image.title="CALO-RPD Studio" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    QT_QPA_PLATFORM=xcb \
    DISPLAY=:99 \
    HOME=/data/home/calo \
    XDG_CACHE_HOME=/tmp/calo-cache \
    XDG_CONFIG_HOME=/data/home/calo/.config \
    XDG_RUNTIME_DIR=/tmp/calo-runtime \
    PYTHONPATH=/opt/calo \
    CALO_CONTAINER_PORT=6080 \
    CALO_WORKDIR=/data

COPY containers/debian.sources /etc/apt/sources.list.d/debian.sources
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        dbus-x11 \
        fonts-dejavu-core \
        libdbus-1-3 \
        libegl1 \
        libgl1 \
        libx11-xcb1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxkbcommon-x11-0 \
        novnc \
        openbox \
        websockify \
        x11vnc \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/calo
COPY requirements-lock-cpu-py311-linux.txt requirements-lock-cuda128-py311-linux.txt pyproject.toml README.md LICENSE ./
RUN python -m pip install --require-hashes --requirement "${RUNTIME_LOCK}"

COPY calo_bootstrap ./calo_bootstrap
COPY calo_rpd_studio ./calo_rpd_studio
COPY bootstrap.py ./bootstrap.py
COPY containers/entrypoint.py /usr/local/bin/calo-container-entrypoint.py
RUN python -m calo_rpd_studio.compute.source_identity \
        --output /opt/calo/.calo-source-identity.json \
        --source-commit "${SOURCE_COMMIT}" \
        --tracked-source-clean "${SOURCE_TRACKED_CLEAN}" \
    && python -m pip check \
    && python -c "import ctypes; from pathlib import Path; from PyQt6.QtCore import QLibraryInfo; plugin = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / 'platforms' / 'libqxcb.so'; ctypes.CDLL(str(plugin)); print(f'Qt xcb dependency closure verified: {plugin}')" \
    && python -m pip uninstall --yes setuptools wheel \
    && python -m pip uninstall --yes pip \
    && groupadd --gid 10001 calo \
    && useradd --uid 10001 --gid 10001 --home-dir /data/home/calo --no-create-home --shell /usr/sbin/nologin calo \
    && mkdir -p /data/home/calo/.config \
    && chown -R 10001:10001 /data

VOLUME ["/data"]
EXPOSE 6080

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD python -c "import os,pathlib,urllib.request; pid=int(pathlib.Path(os.environ.get('CALO_APP_PID_FILE','/tmp/calo-app.pid')).read_text(encoding='ascii')); os.kill(pid,0); urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('CALO_CONTAINER_PORT','6080')+'/vnc.html', timeout=3)" || exit 1

USER 10001:10001
ENTRYPOINT ["python", "/usr/local/bin/calo-container-entrypoint.py"]
