#!/bin/bash

echo "🚀 Iniciando WAN GGUF (Docker Desktop + RTX 3060)..."

# Verificar se Docker Desktop está rodando
if ! docker info &>/dev/null; then
    echo "❌ Docker Desktop não está rodando."
    echo "   Por favor, inicie o Docker Desktop no Windows primeiro."
    exit 1
fi

# Verificar se GPU está disponível no Docker Desktop
echo "🔍 Verificando GPU no Docker Desktop..."
if docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi &>/dev/null; then
    echo "✅ GPU RTX 3060 detectada no Docker Desktop"
    USE_GPU="--profile gpu"
else
    echo "⚠️  GPU não detectada. Verifique:"
    echo "   1. Docker Desktop > Settings > Resources > WSL Integration"
    echo "   2. Docker Desktop > Settings > General > Use WSL 2"
    echo "   3. Reinicie o Docker Desktop se necessário"
    echo ""
    echo "🔄 Executando sem GPU por enquanto..."
    USE_GPU=""
fi

# Verificar diretórios
if [ ! -d "/mnt/e/Docker/wan/wan-gguf" ]; then
    echo "📁 Criando diretórios de storage..."
    mkdir -p /mnt/e/Docker/wan/wan-gguf/{models,cache}
fi

echo "🔨 Construindo imagem Docker..."
docker-compose build --no-cache

echo "🏃 Iniciando container..."
if [ -n "$USE_GPU" ]; then
    docker-compose $USE_GPU up -d
else
    # Versão fallback sem GPU
    docker-compose up -d
fi

echo ""
echo "✅ WAN GGUF rodando!"
echo "🌐 Interface: http://localhost:7860"
echo "📊 Logs: docker-compose logs -f wan-gguf"
echo "🛑 Parar: docker-compose down"
echo ""
echo "⏳ Primeira execução: aguarde download do modelo (~2-5GB)"
echo "🔧 Storage: E:\\Docker\\wan\\wan-gguf\\"

# Mostrar logs iniciais
sleep 3
echo ""
echo "📋 Logs iniciais:"
docker-compose logs --tail=20 wan-gguf
