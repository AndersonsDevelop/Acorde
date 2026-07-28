#!/usr/bin/env bash
# exit on error
set -o errexit

# Instala dependências do Python padrão
pip install -r requirements.txt

# Cria uma pasta para os binários locais se não existir
mkdir -p ./bin

# Baixa o FFmpeg estático para Linux
X_URL="https://johnvansickle.com"
curl -L $X_URL | tar -xJ --strip-components=1 -C ./bin
