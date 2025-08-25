# Video Pipeline (modular n8n)

## O que é
Arquitetura modular para gerar vídeos curtos com seleção de provedores por etapa (LLM, Imagem, TTS, Vídeo, Storage).  
Você controla tudo via `contracts/run.json` e `config/providers.json`.

## Como usar
1) Configure `.env` (vide `.env.example`).
2) Importe os JSONs em `workflows/` no n8n (File > Import from file).
3) Ajuste `config/providers.json` se quiser trocar modelos/URLs.
4) Edite `contracts/run.json` para cada job e **dispare** o Orquestrador pelo n8n (Webhook ou Execute Workflow).

## Estrutura
- config/providers.json  -> catálogo de provedores
- contracts/run.json     -> contrato do job (o que rodar e com quais provedores)
- prompts/*.json         -> prompts pequenos e versionáveis
- baserow/*.csv          -> CSVs para criar tabelas base (registry, accounts)
- workflows/*.json       -> exports dos workflows n8n (orquestrador e etapas)

## Fluxo
Orquestrador lê o `run.json`, resolve IDs do Baserow por slug (sem ID hardcoded), chama sub-workflows:
- Script (LLM) -> Prompts -> Imagem -> TTS -> Vídeo -> Legendas (mock) -> Persistência

