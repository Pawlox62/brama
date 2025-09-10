#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Użycie: $0 <plik.jpg|png>"; exit 1
fi

IMG="$1"
if [ ! -f "$IMG" ]; then
  echo "Błąd: plik nie istnieje: $IMG"; exit 2
fi

DIR="$(dirname "$(readlink -f "$IMG")")"
BASE="$(basename "$IMG")"

# Montujemy KATALOG z obrazem, a wewnątrz kontenera wskazujemy /data/<oryginalna_nazwa_z_rozszerzeniem>
sudo docker run --rm -it \
  -v "$DIR":/data:ro \
  myopenalpr:local \
  -c eu "/data/$BASE"

