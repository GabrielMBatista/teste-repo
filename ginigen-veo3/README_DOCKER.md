# Docker Setup para VEO3-Free

## Configuração dos Volumes

Este setup utiliza volumes Docker persistentes para armazenar modelos e cache em `E:\Docker\ginigen\`:

- `E:\Docker\ginigen\cache\` - Cache do Hugging Face e modelos
- `E:\Docker\ginigen\torch_cache\` - Cache do PyTorch
- `E:\Docker\ginigen\transformers_cache\` - Cache do Transformers
- `E:\Docker\ginigen\models\` - Modelos baixados
- `E:\Docker\ginigen\tmp\` - Arquivos temporários
- `./app.py` - **App customizado** que sobrescreve o original do container

## Como Executar

### Preparação Inicial

Primeiro, crie os diretórios de cache manualmente:

```bash
# No Windows
mkdir E:\Docker\ginigen\cache
mkdir E:\Docker\ginigen\torch_cache
mkdir E:\Docker\ginigen\pip_cache
mkdir E:\Docker\ginigen\transformers_cache
mkdir E:\Docker\ginigen\models
mkdir E:\Docker\ginigen\tmp

# No WSL/Linux
mkdir -p /mnt/e/Docker/ginigen/{cache,torch_cache,pip_cache,transformers_cache,models,tmp}
```

### Docker Compose

```bash
# Iniciar
docker-compose up

# Executar em background
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar
docker-compose down

# Rebuild se necessário
docker-compose up --build
```

## Estrutura de Diretórios

Após a primeira execução, você terá a seguinte estrutura:
