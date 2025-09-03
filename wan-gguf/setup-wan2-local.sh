#!/bin/bash

echo "🚀 Configurando WAN2 para armazenamento local..."

# Criar estrutura de diretórios
echo "📁 Criando estrutura de diretórios..."
mkdir -p wan2-faster

# Verificar se estamos no diretório correto
if [ ! -f "docker-compose.yml" ]; then
    echo "⚠️  Execute este script no diretório raiz do projeto wan-gguf"
    exit 1
fi

# Criar diretórios de storage
echo "📂 Criando diretórios de storage para WAN2..."
mkdir -p /mnt/e/Docker/wan/wan2-models
mkdir -p /mnt/e/Docker/wan/wan2-cache

# Verificar espaço disponível
echo "💾 Verificando espaço em disco..."
AVAILABLE_SPACE=$(df /mnt/e/Docker/wan/ 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")
AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))

if [ $AVAILABLE_GB -gt 0 ]; then
    echo "✅ Espaço disponível: ${AVAILABLE_GB}GB"
    if [ $AVAILABLE_GB -lt 25 ]; then
        echo "⚠️  Recomendado: pelo menos 25GB livres para WAN2"
    fi
else
    echo "❌ Não foi possível verificar espaço no drive E:"
    echo "   Certifique-se de que o drive E: existe e está acessível"
fi

# Verificar GPU
echo "🎮 Verificando GPU..."
if docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi &>/dev/null 2>&1; then
    echo "✅ GPU RTX 3060 detectada"
else
    echo "⚠️  GPU não detectada - WAN2 funcionará na CPU (muito lento)"
fi

echo ""
echo "✅ Setup WAN2 Local concluído!"
echo ""
echo "🚀 Para executar WAN2:"
echo "   cd wan2-faster"
echo "   ./run.sh"
echo ""
echo "📁 Modelos serão salvos em:"
echo "   E:\\Docker\\wan\\wan2-models\\"
echo ""
echo "💡 Primeira execução irá baixar ~14GB de modelos"
echo "⏱️  Tempo estimado de download: 20-60 minutos (dependendo da internet)"

chmod +x wan2-faster/run.sh 2>/dev/null || echo "⚠️  Lembre-se de tornar run.sh executável"
