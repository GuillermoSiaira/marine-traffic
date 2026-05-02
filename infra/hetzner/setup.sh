#!/bin/bash
# Setup script para servidor Hetzner (Ubuntu 22.04/24.04)
# Ejecutar como root justo después de crear el VPS:
#   ssh root@<IP> 'bash -s' < infra/hetzner/setup.sh

set -euo pipefail

echo "==> Actualizando sistema..."
apt-get update -q && apt-get upgrade -y -q

echo "==> Instalando Docker..."
curl -fsSL https://get.docker.com | sh
systemctl enable docker

echo "==> Instalando git y utilidades..."
apt-get install -y -q git htop

echo "==> Clonando repositorio..."
git clone https://github.com/guillermosiaira/marine-traffic.git /opt/marine-traffic
cd /opt/marine-traffic

echo "==> Creando .env..."
cp .env.example .env
echo ""
echo "IMPORTANTE: edita /opt/marine-traffic/.env con tus keys antes de continuar"
echo "  nano /opt/marine-traffic/.env"
echo ""

echo "==> Construyendo imágenes Docker..."
docker compose build

echo ""
echo "======================================================"
echo " Setup completo. Pasos finales:"
echo "   1. nano /opt/marine-traffic/.env   (pega tus keys)"
echo "   2. docker compose up -d collector  (arranca el colector)"
echo "   3. docker compose logs -f          (ver logs en vivo)"
echo "======================================================"
