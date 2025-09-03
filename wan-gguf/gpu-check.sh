#!/bin/bash

echo "🐳 Verificando Docker Desktop + RTX 3060..."

# 1. Verificar se Docker Desktop está rodando
echo "1️⃣ Verificando Docker Desktop..."
if docker info &>/dev/null; then
    echo "✅ Docker Desktop funcionando"
    echo "   📊 Versão: $(docker --version)"
else
    echo "❌ Docker Desktop não está rodando"
    echo "   Por favor, inicie o Docker Desktop no Windows"
    exit 1
fi

# 2. Verificar runtime NVIDIA
echo ""
echo "2️⃣ Verificando runtime NVIDIA..."
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "✅ Runtime NVIDIA detectado no Docker Desktop"
else
    echo "⚠️  Runtime NVIDIA não detectado"
    echo "   Verifique: Docker Desktop > Settings > General > Use WSL 2 based engine"
fi

# 3. Testar GPU com container simples
echo ""
echo "3️⃣ Testando GPU com container..."
if docker run --rm --gpus all hello-world &>/dev/null; then
    echo "✅ Suporte GPU básico funcionando"
else
    echo "⚠️  Teste básico de GPU falhou"
fi

# 4. Testar CUDA específico
echo ""
echo "4️⃣ Testando CUDA + nvidia-smi..."
echo "   Baixando imagem CUDA (pode demorar na primeira vez)..."

# Testar diferentes versões de CUDA
for cuda_version in "12.0-base-ubuntu20.04" "11.8-base-ubuntu20.04" "latest"; do
    echo "   Testando nvidia/cuda:$cuda_version..."
    if timeout 60 docker run --rm --gpus all nvidia/cuda:$cuda_version nvidia-smi 2>/dev/null; then
        echo "✅ GPU funcionando com CUDA $cuda_version"
        echo "   RTX 3060 detectada no Docker!"
        break
    else
        echo "   ❌ Falha com $cuda_version"
    fi
done

# 5. Verificar diretórios de storage
echo ""
echo "5️⃣ Verificando storage no Windows..."
STORAGE_PATH="/mnt/e/Docker/wan/wan-gguf"
WIN_STORAGE_PATH="E:\\Docker\\wan\\wan-gguf"

if [ -d "$STORAGE_PATH" ] || mkdir -p "$STORAGE_PATH" 2>/dev/null; then
    echo "✅ Storage WSL: $STORAGE_PATH"
    echo "✅ Storage Windows: $WIN_STORAGE_PATH"
    
    # Teste de escrita
    if echo "test" > "$STORAGE_PATH/test.txt" 2>/dev/null; then
        rm "$STORAGE_PATH/test.txt"
        echo "✅ Permissões de escrita OK"
    else
        echo "⚠️  Problema de permissões de escrita"
    fi
else
    echo "❌ Não conseguiu criar diretório de storage"
    echo "   Verifique se o drive E: existe no Windows"
fi

# 6. Informações do sistema
echo ""
echo "6️⃣ Informações do sistema:"
echo "   💻 WSL: $(cat /proc/version | grep -o 'WSL[0-9]*')"
echo "   🐳 Docker: $(docker --version)"
echo "   🎮 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Não detectada via nvidia-smi')"

# 7. Recomendações
echo ""
echo "📋 Configurações recomendadas no Docker Desktop:"
echo "   • Settings > General > Use WSL 2 based engine ✓"
echo "   • Settings > Resources > WSL Integration > Ubuntu ✓"
echo "   • Settings > Docker Engine > Adicionar runtime nvidia"
echo ""
echo "🚀 Se tudo estiver ✅, você pode executar:"
echo "   ./run.sh"