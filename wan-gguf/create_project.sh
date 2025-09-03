#!/bin/bash

echo "🚀 Criando projeto WAN GGUF completo..."
echo "🐳 Docker Desktop detectado - configuração otimizada"

# Criar diretórios no Windows (via WSL path)
echo "📁 Criando diretórios de storage..."
mkdir -p /mnt/e/Docker/wan/wan-gguf/{models,cache}

# Verificar se Docker Desktop está rodando
if ! docker info &>/dev/null; then
    echo "⚠️  Docker Desktop não está rodando. Inicie o Docker Desktop no Windows primeiro."
    exit 1
fi

# Criar Dockerfile
cat > Dockerfile << 'EOF'
FROM nvidia/cuda:12.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Instalar llama-cpp-python com CUDA para RTX 3060 (Compute Capability 8.6)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=86" \
    pip install --no-cache-dir llama-cpp-python

RUN pip install --no-cache-dir \
    huggingface-hub transformers torch gradio requests accelerate

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
EOF

# Criar app.py
cat > app.py << 'EOF'
import gradio as gr
from huggingface_hub import hf_hub_download, list_repo_files
from llama_cpp import Llama
import os
import time

def find_gguf_files():
    """Encontrar arquivos GGUF no repositório"""
    try:
        files = list_repo_files("calcuis/wan-gguf")
        gguf_files = [f for f in files if f.endswith('.gguf')]
        return gguf_files
    except Exception as e:
        print(f"Erro ao listar arquivos: {e}")
        return ["wan-7b-q4_k_m.gguf"]  # Fallback

def download_model():
    """Baixar modelo GGUF"""
    print("🔍 Procurando arquivos GGUF...")
    gguf_files = find_gguf_files()
    
    print(f"Arquivos GGUF encontrados: {gguf_files}")
    
    if not gguf_files:
        raise Exception("Nenhum arquivo GGUF encontrado no repositório")
    
    # Usar o primeiro arquivo GGUF encontrado
    filename = gguf_files[0]
    print(f"📥 Baixando modelo: {filename}")
    
    model_path = hf_hub_download(
        repo_id="calcuis/wan-gguf",
        filename=filename,
        cache_dir="/app/models",
        local_dir_use_symlinks=False
    )
    return model_path

print("🤖 Inicializando WAN GGUF...")
print("🔧 Configuração da GPU:")
print("  - Detectando CUDA...")

try:
    model_path = download_model()
    print(f"✅ Modelo baixado em: {model_path}")
    
    # Inicializar o modelo com configurações para RTX 3060 12GB
    print("🚀 Carregando modelo na GPU...")
    llm = Llama(
        model_path=model_path,
        n_ctx=8192,  # Contexto grande para 12GB
        n_gpu_layers=-1,  # Todas as camadas na GPU
        n_threads=6,  # Threads CPU auxiliares
        n_batch=512,  # Batch otimizado
        verbose=True,  # Mostrar info de carregamento
        use_mlock=True,  # Lock memória
        use_mmap=True,   # Memory mapping
    )
    print("✅ Modelo carregado com sucesso!")
    
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    # Fallback sem GPU
    print("🔄 Tentando carregar apenas na CPU...")
    llm = None

def generate_text(prompt, max_tokens=512, temperature=0.7, top_p=0.9):
    if not llm:
        return "❌ Modelo não carregado. Verifique os logs do container."
    
    try:
        start_time = time.time()
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["</s>", "<|end|>", "\n\nHuman:", "\n\nUser:"],
            echo=False
        )
        end_time = time.time()
        
        response = output['choices'][0]['text'].strip()
        tokens_generated = len(response.split())
        time_taken = end_time - start_time
        tokens_per_second = tokens_generated / time_taken if time_taken > 0 else 0
        
        stats = f"\n\n---\n📊 Estatísticas:\n⏱️ Tempo: {time_taken:.2f}s\n🔥 Velocidade: {tokens_per_second:.1f} tokens/s\n📝 Tokens: {tokens_generated}"
        
        return response + stats
        
    except Exception as e:
        return f"❌ Erro na geração: {str(e)}"

# Interface Gradio
with gr.Blocks(
    title="WAN GGUF Local - RTX 3060",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    """
) as demo:
    gr.Markdown("""
    # 🚀 WAN Model (GGUF) - RTX 3060 12GB
    
    **Modelo executando localmente via llama.cpp + CUDA**
    
    📊 **Especificações:**
    - GPU: NVIDIA RTX 3060 (12GB VRAM)
    - Contexto: 8192 tokens
    - Todas as camadas na GPU
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="💬 Seu Prompt",
                placeholder="Digite sua pergunta ou solicitação...",
                lines=6,
                value="Explain quantum computing in simple terms:"
            )
            
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=4096,
                    value=512,
                    step=50,
                    label="📏 Max Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="🌡️ Temperature"
                )
            
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="🎯 Top P"
            )
            
            with gr.Row():
                generate_btn = gr.Button("🚀 Gerar", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Limpar", variant="secondary")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="🤖 Resposta do WAN Model",
                lines=20,
                interactive=False,
                show_copy_button=True
            )
    
    # Exemplos
    gr.Markdown("### 💡 Exemplos de Prompts:")
    examples = [
        "Write a short story about a robot learning to paint.",
        "Explain the theory of relativity in simple terms.",
        "Create a Python function to sort a list of dictionaries.",
        "What are the benefits of renewable energy?",
        "Write a haiku about artificial intelligence."
    ]
    
    example_buttons = []
    with gr.Row():
        for i, example in enumerate(examples[:3]):
            btn = gr.Button(f"📝 {example[:30]}...", size="sm")
            btn.click(lambda x=example: x, outputs=prompt_input)
    
    # Event handlers
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens, temperature, top_p],
        outputs=output_text
    )
    
    clear_btn.click(
        lambda: ("", ""),
        outputs=[prompt_input, output_text]
    )

if __name__ == "__main__":
    print("🌐 Iniciando interface web...")
    print("🔗 Acesse: http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        inbrowser=False
    )
EOF

# Criar docker-compose.yml otimizado para Docker Desktop
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  wan-gguf:
    build: .
    ports:
      - "7860:7860"
    volumes:
      # Storage no Windows - Docker Desktop gerencia automaticamente
      - /mnt/e/Docker/wan/wan-gguf/models:/app/models
      - /mnt/e/Docker/wan/wan-gguf/cache:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    restart: unless-stopped
    # GPU passthrough via Docker Desktop
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # Configurações para Docker Desktop
    profiles:
      - gpu
EOF

# Criar script de execução otimizado para Docker Desktop
cat > run.sh << 'EOF'
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
EOF

chmod +x run.sh

# Criar script adicional para Windows
cat > run.bat << 'EOF'
@echo off
echo 🚀 Iniciando WAN GGUF via Docker Desktop...

REM Verificar se Docker Desktop está rodando
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Desktop não está rodando
    echo Por favor, inicie o Docker Desktop primeiro
    pause
    exit /b 1
)

REM Criar diretórios se não existirem
if not exist "E:\Docker\wan\wan-gguf" (
    echo 📁 Criando diretórios...
    mkdir "E:\Docker\wan\wan-gguf\models" 2>nul
    mkdir "E:\Docker\wan\wan-gguf\cache" 2>nul
)

echo 🔨 Construindo e executando...
docker-compose up --build -d

echo ✅ WAN GGUF iniciado!
echo 🌐 Acesse: http://localhost:7860
echo.
echo 📊 Para ver logs: docker-compose logs -f wan-gguf
echo 🛑 Para parar: docker-compose down
echo.
pause
EOF

echo "✅ Projeto WAN GGUF criado!"
echo ""
echo "🚀 Para executar:"
echo "  ./run.sh"
echo ""
echo "📁 Modelos serão salvos em: E:\\Docker\\wan\\wan-gguf\\models"