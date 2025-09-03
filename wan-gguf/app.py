import gradio as gr
import os
import time
import torch
import numpy as np
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
from diffusers.utils import export_to_video
import warnings
warnings.filterwarnings("ignore")


def check_gpu_availability():
    """Verificar disponibilidade da GPU"""
    print("🔧 Verificando GPU...")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"✅ CUDA disponível!")
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 VRAM Total: {gpu_memory:.1f}GB")
        print(f"🔢 GPUs disponíveis: {gpu_count}")

        # Testar alocação GPU
        try:
            test_tensor = torch.randn(100, 100).cuda()
            print(f"🧪 Teste de alocação GPU: ✅")
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"❌ Falha no teste de GPU: {e}")
            return False
    else:
        print("❌ CUDA não disponível - usando CPU")
        return False


def load_wan_model():
    """Carregar modelo WAN usando diffusers (forma correta)"""

    # Verificar GPU primeiro
    gpu_available = check_gpu_availability()
    device = "cuda" if gpu_available else "cpu"

    print(f"\n📥 Carregando WAN Model via diffusers...")
    print(f"📂 Repositório: Comfy-Org/Wan_2.1_ComfyUI_repackaged")
    print(f"🎯 Dispositivo alvo: {device}")
    print(f"💡 Usando diffusers (método correto para WAN)")

    try:
        # Tentar carregar como pipeline de diffusers
        print("🔧 Tentando carregar como DiffusionPipeline...")

        pipeline = DiffusionPipeline.from_pretrained(
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            cache_dir="/app/models",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )

        if device == "cuda":
            pipeline = pipeline.to("cuda")
            print("✅ Pipeline movido para GPU")

        print(f"✅ WAN Model carregado via diffusers no {device.upper()}!")
        return pipeline, f"WAN 2.1 (Diffusers - {device.upper()})", device, "diffusers"

    except Exception as e:
        print(f"❌ Erro ao carregar via diffusers: {e}")

        # Tentar como AutoPipeline
        try:
            print("🔄 Tentando AutoPipelineForText2Image...")

            pipeline = AutoPipelineForText2Image.from_pretrained(
                "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                cache_dir="/app/models",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )

            if device == "cuda":
                pipeline = pipeline.to("cuda")

            print(
                f"✅ WAN Model carregado via AutoPipeline no {device.upper()}!")
            return pipeline, f"WAN 2.1 (AutoPipeline - {device.upper()})", device, "auto"

        except Exception as e2:
            print(f"❌ Erro no AutoPipeline: {e2}")

            # Verificar se é realmente um modelo de texto
            try:
                print("🔄 Tentando carregar componentes individuais...")
                from transformers import AutoTokenizer, AutoModelForCausalLM

                # Tentar carregar como modelo de texto tradicional
                tokenizer = AutoTokenizer.from_pretrained(
                    "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                    cache_dir="/app/models",
                    trust_remote_code=True
                )

                model = AutoModelForCausalLM.from_pretrained(
                    "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                    cache_dir="/app/models",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True
                )

                from transformers import pipeline
                text_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )

                print(
                    f"✅ WAN carregado como modelo de texto no {device.upper()}!")
                return text_pipeline, f"WAN 2.1 (Text - {device.upper()})", device, "text"

            except Exception as e3:
                print(f"❌ Falha final: {e3}")
                return None, "Erro no carregamento", "cpu", "none"


print("🤖 Inicializando WAN Model Otimizado...")
print("🎯 Objetivo: Usar modelo WAN específico na RTX 3060")
print("🔧 Método: diffusers (forma correta para WAN)")

model_description = "Não carregado"
device_info = "cpu"
model_type = "none"
pipeline = None

try:
    pipeline, model_description, device_info, model_type = load_wan_model()

except Exception as e:
    print(f"❌ Erro geral no carregamento: {e}")
    pipeline = None


def generate_content(prompt, max_tokens=512, temperature=0.7, top_p=0.9):
    if not pipeline:
        return f"""❌ Modelo WAN não carregado.

🔍 **Diagnóstico:**
- Falha ao carregar WAN via diffusers
- Modelo pode não estar no formato esperado
- Verificar se repositório é realmente um modelo WAN funcional

💡 **Soluções:**
1. Verificar se repositório tem modelo WAN válido
2. Usar ComfyUI para modelos WAN T2V específicos
3. Tentar repositório alternativo do WAN

📊 **Status:** {model_description}
🔧 **Dispositivo:** {device_info}
🎭 **Tipo:** {model_type}
"""

    try:
        start_time = time.time()

        # Verificar se está realmente usando GPU
        if device_info == "cuda":
            print(f"🎮 Gerando com WAN na GPU...")
            torch.cuda.empty_cache()
        else:
            print(f"💻 Gerando com WAN na CPU...")

        # Geração baseada no tipo de modelo
        if model_type == "diffusers" or model_type == "auto":
            # Para modelos de difusão (geração de imagem/vídeo)
            result = pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            )

            if hasattr(result, 'images'):
                response = f"✅ Imagem gerada com WAN Model!\n🖼️ Resolução: 512x512\n📝 Prompt: {prompt}"
            else:
                response = f"✅ Conteúdo gerado com WAN Model!\n📝 Prompt: {prompt}"

        elif model_type == "text":
            # Para modelos de texto
            generation_config = {
                "max_new_tokens": min(max_tokens, 512),
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "return_full_text": False
            }

            result = pipeline(prompt, **generation_config)

            if isinstance(result, list) and len(result) > 0:
                response = result[0]["generated_text"].strip()
            else:
                response = "Erro na extração do texto gerado"
        else:
            response = "Tipo de modelo não suportado"

        end_time = time.time()

        # Calcular estatísticas
        time_taken = end_time - start_time

        # Adicionar info da GPU
        gpu_status = ""
        if device_info == "cuda":
            try:
                gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                gpu_memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                gpu_status = f"\n🎮 GPU Usado: {gpu_memory_used:.1f}GB | Cache: {gpu_memory_cached:.1f}GB"
            except:
                gpu_status = "\n🎮 GPU: Ativo"

        stats = f"\n\n---\n📊 **Estatísticas WAN:**\n⏱️ Tempo: {time_taken:.2f}s\n🤖 Modelo: {model_description}\n🔧 Dispositivo: {device_info.upper()}\n🎭 Tipo: {model_type.upper()}{gpu_status}"

        # Limpar cache GPU após geração
        if device_info == "cuda":
            torch.cuda.empty_cache()

        return response + stats

    except Exception as e:
        return f"❌ Erro na geração WAN: {str(e)}\n\n🤖 Modelo: {model_description}\n🔧 Dispositivo: {device_info}"


# Interface Gradio
with gr.Blocks(
    title="WAN Model Otimizado - RTX 3060",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    """
) as demo:
    gr.Markdown(f"""
    # 🚀 WAN Model Otimizado - RTX 3060 12GB
    
    **Modelo WAN executando via diffusers/transformers**
    
    📊 **Especificações:**
    - GPU: NVIDIA RTX 3060 (12GB VRAM)
    - Modelo: {model_description}
    - Backend: diffusers/transformers
    - Dispositivo: {device_info}
    - Tipo: {model_type}
    
    ✅ **Repositório:** [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="💬 Prompt para WAN Model",
                placeholder="Digite seu prompt para o modelo WAN...",
                lines=6,
                value="A futuristic city with flying cars at sunset"
            )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=1024,
                    value=256,
                    step=50,
                    label="📏 Max Tokens (para texto)"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
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
                generate_btn = gr.Button(
                    "🚀 Gerar com WAN", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Limpar", variant="secondary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="🤖 Resultado do WAN Model",
                lines=20,
                interactive=False,
                show_copy_button=True
            )

    # Exemplos específicos para WAN
    gr.Markdown("### 💡 Exemplos de Prompts para WAN:")
    examples = [
        "A robot walking through a neon-lit cyberpunk street",
        "Ocean waves crashing against cliffs during a storm",
        "A magical forest with glowing mushrooms and fireflies",
        "Futuristic spacecraft landing on an alien planet",
        "Abstract geometric patterns morphing and flowing"
    ]

    with gr.Row():
        for i, example in enumerate(examples[:3]):
            btn = gr.Button(f"📝 {example[:25]}...", size="sm")
            btn.click(lambda x=example: x, outputs=prompt_input)

    # Event handlers
    generate_btn.click(
        fn=generate_content,
        inputs=[prompt_input, max_tokens, temperature, top_p],
        outputs=output_text
    )

    clear_btn.click(
        lambda: ("", ""),
        outputs=[prompt_input, output_text]
    )

if __name__ == "__main__":
    print("🌐 Iniciando interface WAN...")
    port = int(os.environ.get('GRADIO_SERVER_PORT', 7861))
    print(f"🔗 Acesse: http://localhost:{port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_api=False,
        inbrowser=False
    )
