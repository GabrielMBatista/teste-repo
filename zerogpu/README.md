# WAN2 Model Local Setup

Este setup permite rodar o modelo WAN2 localmente com arquivos pesados armazenados fora do container Docker.

## Pré-requisitos

- Docker Desktop com suporte a GPU
- WSL2 com Ubuntu/Debian
- Python 3.8+
- Git LFS
- Pelo menos 10GB de espaço livre em E:\Docker\wan

## Configuração WSL

1. **Configurar acesso ao disco E: no WSL:**

   ```bash
   chmod +x setup_wsl.sh
   ./setup_wsl.sh
   ```

2. **Primeira execução:**

   ```bash
   chmod +x run.sh
   ./run.sh
   ```

3. **Execuções subsequentes:**
   ```bash
   docker-compose up
   ```

## Estrutura de Arquivos

```
E:\Docker\wan\          # Modelos (acessível via /mnt/e/Docker/wan no WSL)
├── model.bin
├── tokenizer.json
└── outros arquivos...

/home/gabriel/portifolio/teste-repo/zerogpu/
├── docker-compose.yml  # Configuração principal
├── Dockerfile.local    # Dockerfile customizado
├── download_models.py  # Script de download
├── setup_wsl.sh       # Configuração WSL
└── run.sh             # Script de execução
```

## Dica de Otimização

Para evitar gargalos de I/O ao carregar os modelos, copie os arquivos do modelo para o diretório temporário `/tmp/models_temp` dentro do container antes de iniciar o app:

```bash
docker exec -it <container_id> bash -c "cp -r /tmp/models /tmp/models_temp"
```

Isso garante que os arquivos sejam acessados diretamente do disco do container, reduzindo o tempo de carregamento.

## Troubleshooting

- **Erro de montagem WSL:** Execute `sudo mount -t drvfs E: /mnt/e`
- **Permissões negadas:** Verifique se o WSL tem acesso ao disco E:
- **Out of Memory:** Os arquivos pesados ficam no disco E:, não na memória do Docker
- **GPU não detectada:** Verificar se Docker Desktop tem suporte a GPU habilitado

## Comandos Úteis WSL

```bash
# Verificar se E: está montado
mountpoint /mnt/e

# Montar manualmente
sudo mount -t drvfs E: /mnt/e

# Listar arquivos no diretório Windows
ls -la /mnt/e/Docker/wan
```
