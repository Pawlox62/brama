#!/usr/bin/env bash
set -euo pipefail

IMAGE="pawloxdocker/brama:latest"   # obraz aplikacji (gate_app)
APP_NAME="brama"
CONTAINER_NAME="${APP_NAME}_app"
SERVICE_NAME="brama.service"

PROJECT_DIR="$(pwd)"
ODCZYTY_DIR="${PROJECT_DIR}/odczyty"
ANPR_LOG="${PROJECT_DIR}/anpr.log"
ALPR_TMP_HOST="${PROJECT_DIR}/alprtmp"   # współdzielony katalog ROI na hoście
TEST_FRAMES_DIR="${PROJECT_DIR}/test_frames"

USER_UID=$(id -u)
USER_GID=$(id -g)

msg(){ echo -e "\e[32m[*]\e[0m $*"; }

# --- Instalacja Dockera (jeśli brak) ---
if ! command -v docker >/dev/null 2>&1; then
  msg "Instaluję Dockera…"
  sudo apt-get update -y
  sudo apt-get install -y ca-certificates curl gnupg lsb-release
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/$(. /etc/os-release; echo "$ID")/gpg" | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$(. /etc/os-release; echo "$ID") $(. /etc/os-release; echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo systemctl enable --now docker
else
  msg "Docker już zainstalowany."
fi

# --- Katalogi i pliki ---
mkdir -p "${ODCZYTY_DIR}" "${ALPR_TMP_HOST}" "${PROJECT_DIR}/db"
mkdir -p "${TEST_FRAMES_DIR}/roi" "${TEST_FRAMES_DIR}/full" "${TEST_FRAMES_DIR}/blur" "${TEST_FRAMES_DIR}/noresult" "${TEST_FRAMES_DIR}/fullstream"
touch "${ANPR_LOG}"

if [ ! -f "${PROJECT_DIR}/config.yaml" ]; then
cat > "${PROJECT_DIR}/config.yaml" <<'YAML'
RTSP_URL: "rtsp://192.168.0.100:554/Streaming/Channels/101"
GATE_URL: "https://example.org/direct/xxx/open-close"

# Przetwarzanie
PROCESS_EVERY_N_FRAME: 3
CONFIRM_FRAMES: 2
CONFIRM_WINDOW_SEC: 2.0
COOLDOWN_SECONDS: 20
CROP_PAD: 6
PLATE_REGEX: "^[A-Z0-9]{4,10}$"

# OpenALPR (Docker)
DOCKER_IMAGE: "pawloxdocker/alpr:latest"
DOCKER_USE_SUDO: false
ALPR_CONTAINER_NAME: "alprd"
ALPR_COUNTRY: "eu"
ALPR_CONFIDENCE_MIN: 85.0

# Whitelist
WHITELIST_FILE: "./whitelist.txt"
WHITELIST_RELOAD_SEC: 15

# Logi
LOG_PATH: "./anpr.log"
HEARTBEAT_SEC: 30
HEALTHCHECK: false

# Zapisy pełnych klatek dla tej samej tablicy
SAVE_MIN_INTERVAL_SEC: 120
SAVE_MIN_HASH_DIFF: 6

# ROI (większe na start)
ROI_REL_W: 0.65
ROI_REL_H: 0.45

# Debug: zapisuj wszystkie klatki testowe
SAVE_ALL_TEST_FRAMES: true
TEST_FRAMES_DIR: "./test_frames"

# Opcjonalny sampling pełnego strumienia (0 = off)
FULLSTREAM_DUMP_EVERY_SEC: 0
YAML
  msg "Utworzono domyślny config.yaml – uzupełnij RTSP_URL i GATE_URL."
fi

if [ ! -f "${PROJECT_DIR}/whitelist.txt" ]; then
cat > "${PROJECT_DIR}/whitelist.txt" <<'TXT'
# TABLICA[, Etykieta]
WX12345, Mieszkaniec Kowalski
TXT
  msg "Utworzono przykładowy whitelist.txt."
fi

# --- Pobranie obrazu aplikacji ---
msg "Pobieram obraz Dockera aplikacji: ${IMAGE}"
sudo docker pull "${IMAGE}"

# --- Unit systemd (brama.service) ---
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"
sudo bash -c "cat > '${SERVICE_FILE}'" <<EOF
[Unit]
Description=Brama ANPR (Docker)
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=simple
TimeoutStartSec=0
WorkingDirectory=${PROJECT_DIR}
Environment=IMAGE=${IMAGE}
Environment=CONTAINER=${CONTAINER_NAME}
Environment=PROJECT_DIR=${PROJECT_DIR}
Environment=ODCZYTY_DIR=${ODCZYTY_DIR}
Environment=ANPR_LOG=${ANPR_LOG}
Environment=ALPR_TMP_HOST=${ALPR_TMP_HOST}
Environment=TZ=Europe/Warsaw
Environment=RUN_UID=${USER_UID}
Environment=RUN_GID=${USER_GID}

# zawsze pobierz najnowszy obraz aplikacji i usuń poprzedni kontener
ExecStartPre=/usr/bin/docker pull \${IMAGE}
ExecStartPre=-/usr/bin/docker rm -f \${CONTAINER}

# uruchom aplikację (gate_app), przekazując HOST_ALPR_TMP oraz montując katalogi
ExecStart=/usr/bin/docker run --name \${CONTAINER} --network host \
  --restart unless-stopped \
  -v \${PROJECT_DIR}/config.yaml:/app/config.yaml:ro \
  -v \${PROJECT_DIR}/whitelist.txt:/app/whitelist.txt:ro \
  -v \${ODCZYTY_DIR}:/app/odczyty \
  -v \${ANPR_LOG}:/app/anpr.log \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v \${ALPR_TMP_HOST}:/shared/alprtmp \
  -v \${PROJECT_DIR}/db:/app/db \
  -v \${PROJECT_DIR}/test_frames:/app/test_frames \
  -e TZ=\${TZ} \
  -e HOST_ALPR_TMP=\${ALPR_TMP_HOST} \
  \${IMAGE}

ExecStop=/usr/bin/docker stop -t 10 \${CONTAINER}
ExecStopPost=-/usr/bin/docker rm -f \${CONTAINER}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"

# --- Alias installer (TYLKO brama) ---
install_aliases() {
  local BASHRC="$HOME/.bashrc"
  local START_MARK="# >>> brama aliases >>>"
  local END_MARK="# <<< brama aliases <<<"

  if [ -f "$BASHRC" ] && grep -qF "$START_MARK" "$BASHRC"; then
    sed -i "/$START_MARK/,/$END_MARK/d" "$BASHRC"
  fi

  cat >> "$BASHRC" <<'ALIASBLOCK'
# >>> brama aliases >>>
alias sbrama='sudo systemctl start brama.service'
alias rbrama='sudo systemctl restart brama.service'
alias kbrama='sudo systemctl stop brama.service'
alias logbrama='journalctl -u brama.service -f -o cat'
# <<< brama aliases <<<
ALIASBLOCK

  # natychmiastowe wczytanie aliasów do bieżącej sesji
  # shellcheck disable=SC1090
  source "$BASHRC"
}

install_aliases

echo ""
echo "======================================================"
msg "Instalacja BRAMY zakończona ✅"
echo ""
echo "📂 Katalog projektu:  ${PROJECT_DIR}"
echo "⚙️  Edytuj konfigurację:  ${PROJECT_DIR}/config.yaml (RTSP_URL, GATE_URL)"
echo ""
echo "➡️  Najważniejsze aliasy:"
echo "   ▶ sbrama   – start usługi bramy"
echo "   ▶ rbrama   – restart usługi bramy"
echo "   ▶ kbrama   – zatrzymanie bramy"
echo "   ▶ logbrama – podgląd logów bramy"
echo "======================================================"

