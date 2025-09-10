# --- APP: gate_app (uruchamia gate_app.py) ---
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Warsaw

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg tzdata \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
      ca-certificates curl gnupg \
 && rm -rf /var/lib/apt/lists/*

# docker-ce-cli (klient dockera) – bez demona
RUN mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
 && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY gate_app.py /app/gate_app.py
COPY config.yaml /app/config.yaml
COPY whitelist.txt /app/whitelist.txt

RUN pip install --no-cache-dir opencv-python-headless numpy flask requests pyyaml

ENV LD_LIBRARY_PATH="/usr/local/lib"
ENV ALPR_TMP_DIR="/shared/alprtmp"

CMD ["python", "gate_app.py"]

