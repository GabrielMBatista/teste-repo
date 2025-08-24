# Sistema Criador de Vídeos

Repositório de exemplo para automatizar a criação de vídeos curtos utilizando n8n, Baserow, MinIO, NCA Toolkit e Kokoro TTS.

## Pré-requisitos
- [Docker Compose](https://docs.docker.com/compose/) instalado.

## Configuração
1. Copie o arquivo de variáveis e edite conforme necessário:
   ```bash
   cp env/.env.example .env
   # edite o arquivo .env com suas chaves e URLs
   ```
   Opcionalmente, execute o front-end e preencha o formulário em `http://localhost:3000`:
   ```bash
   node webfront/server.js
   ```

## Inicialização
Dentro do diretório `docker`, suba a stack completa:
```bash
cd docker
docker compose up -d
```

## Importação de dados
- No n8n (http://localhost:5678), importe o fluxo `../n8n-workflows/sistema_criador_de_videos.json`.
- No Baserow (http://localhost:85), importe `../baserow/tabelas-iniciais.csv` para criar as tabelas iniciais.

## Requisitos de hardware
Para melhor desempenho recomenda-se pelo menos 4 GB de RAM. O serviço `kokoro-tts` utiliza uma imagem GPU por padrão; caso não possua GPU, altere para a imagem de CPU mencionada no `docker-compose`.

## Licença
Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.
