FROM docker.n8n.io/n8nio/n8n:latest

USER root

# (Opcional) Instala o ffmpeg, se ainda necessário
RUN apk add --no-cache ffmpeg

USER node
