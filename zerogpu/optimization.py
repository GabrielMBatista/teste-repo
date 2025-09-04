import torch
import gc
from PIL import Image

# Copiar o arquivo optimization.py original do container
# Este arquivo será substituído pelo original durante o build


def optimize_pipeline_(pipe, **kwargs):
    """
    Função de otimização mais conservadora para evitar OOM
    """
    print("Optimizing pipeline...")

    try:
        # Limpar cache antes da otimização
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Detectar device disponível
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Detectar o device real do modelo pipeline
        try:
            model_device = next(pipe.transformer.parameters()).device
        except:
            model_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        print(f"Modelo está no device: {model_device}")

        # Usar configurações mínimas para otimização
        test_kwargs = {
            'image': kwargs.get('image', Image.new('RGB', (512, 512))),
            'prompt': kwargs.get('prompt', 'test'),
            'height': 512,
            'width': 512,
            'num_frames': 8,
            'num_inference_steps': 1,
            'guidance_scale': 1.0,
            'guidance_scale_2': 1.0,
            'generator': torch.Generator(device=model_device).manual_seed(42),
        }

        print("Executando geração de teste para otimização...")
        with torch.no_grad():
            _ = pipe(**test_kwargs)

        print("Pipeline otimizado com sucesso!")

    except Exception as e:
        print(f"Otimização falhou: {e}")
        print("Continuando sem otimização...")

    # Limpar memória após otimização
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Otimização concluída!")
