# GPU Memory Usage Breakdown

## Por que apenas 0.25GB no início?

### Componentes do Pipeline de Vídeo

1. **VAE (Variational Autoencoder)**: ~0.25GB

   - Responsável por codificar/decodificar imagens
   - Relativamente pequeno comparado aos transformers

2. **Transformer Principal**: ~8-10GB

   - O modelo maior do pipeline
   - Processa a geração de vídeo frame por frame
   - É onde a maior parte da memória será usada

3. **Text Encoder**: ~1-2GB
   - Processa prompts de texto
   - Carregado junto com o transformer

### Carregamento Sequencial

O carregamento está acontecendo assim:
