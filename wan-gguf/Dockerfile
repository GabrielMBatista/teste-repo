FROM python:3.9-slim

# Evitar prompts interativos durante instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Atualizar pip
RUN python -m pip install --upgrade pip

WORKDIR /app

# Instalar PyTorch com CUDA para RTX 3060 (versão compatível)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar diffusers e dependências específicas para WAN
RUN pip install --no-cache-dir \
    diffusers[torch] \
    transformers \
    accelerate \
    huggingface-hub \
    gradio \
    requests \
    numpy \
    sentencepiece \
    protobuf \
    safetensors \
    tokenizers \
    pillow \
    opencv-python-headless

COPY app.py .

EXPOSE 7861

CMD ["python", "app.py"]
