# PyTorch 2.8 (temporary hack)
import shutil
from datetime import datetime
from optimization import optimize_pipeline_
import gc
import random
from PIL import Image
import numpy as np
import tempfile
import gradio as gr
from diffusers.utils.export_utils import export_to_video
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
import torch
import spaces
import os

# Configurar todos os caches para usar disco local ANTES de importar qualquer coisa
os.environ['HF_HOME'] = '/root/.cache/huggingface'
os.environ['TORCH_HOME'] = '/root/.cache/torch'
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/transformers'
os.environ['HF_DATASETS_CACHE'] = '/root/.cache/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface'
os.environ['XDG_CACHE_HOME'] = '/root/.cache'
os.environ['TMPDIR'] = '/tmp/cache'

# Configurar diretórios para salvar arquivos no disco local
OUTPUT_DIR = "/tmp/models/outputs"
MODELS_DIR = "/tmp/models"
CACHE_DIR = "/tmp/cache"

# Criar todos os diretórios necessários
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs('/root/.cache/huggingface', exist_ok=True)
os.makedirs('/root/.cache/torch', exist_ok=True)
os.makedirs('/root/.cache/pip', exist_ok=True)

print(f"Diretório de saída configurado: {OUTPUT_DIR}")
print(f"Diretório de modelos: {MODELS_DIR}")
print(f"Cache HuggingFace: {os.environ.get('HF_HOME')}")
print(f"Cache PyTorch: {os.environ.get('TORCH_HOME')}")

os.system('pip install --upgrade --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu126 "torch<2.9" spaces safetensors')

# Copiar arquivos do modelo para um diretório temporário no container
TEMP_MODEL_DIR = "/tmp/models_temp"
os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

print(f"Copiando arquivos do modelo para {TEMP_MODEL_DIR}...")
shutil.copytree(MODELS_DIR, TEMP_MODEL_DIR, dirs_exist_ok=True)
print("Arquivos copiados com sucesso!")

# Atualizar o cache_dir para o diretório temporário
CACHE_DIR = TEMP_MODEL_DIR

# Actual demo code

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81

MIN_DURATION = round(MIN_FRAMES_MODEL/FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL/FIXED_FPS, 1)

default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"


def resize_image(image: Image.Image) -> Image.Image:
    """Resizes an image to fit within the model's constraints"""
    width, height = image.size

    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)

    aspect_ratio = width / height

    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM

    image_to_resize = image

    if aspect_ratio > MAX_ASPECT_RATIO:
        target_w, target_h = MAX_DIM, MIN_DIM
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        target_w, target_h = MIN_DIM, MAX_DIM
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))
    else:
        if width > height:
            target_w = MAX_DIM
            target_h = int(round(target_w / aspect_ratio))
        else:
            target_h = MAX_DIM
            target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF

    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))

    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)


# Verificar disponibilidade da GPU antes de carregar modelos
print("Verificando disponibilidade da GPU...")

if not torch.cuda.is_available():
    print("ERRO: CUDA não está disponível!")
    print("Dispositivos disponíveis:", torch.cuda.device_count())
    print("Versão PyTorch:", torch.__version__)
    print("Versão CUDA compilada:", torch.version.cuda)
    device = "cpu"
    print("AVISO: Usando CPU - performance será muito lenta!")
else:
    device = "cuda"
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(
        f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(
        f"Memória GPU livre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB")

# Carregar modelo com verificação de dispositivo
print("Carregando modelos...")

# Limpar cache GPU antes de carregar
if device == "cuda":
    gc.collect()
    torch.cuda.empty_cache()

# Criar diretório de offload
offload_dir = os.path.join(CACHE_DIR, "offload")
os.makedirs(offload_dir, exist_ok=True)

try:
    if device == "cuda":
        # Configuração para GPU - usar device_map específico para evitar CPU offload
        print("Carregando modelo para GPU...")
        pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            transformer=WanTransformer3DModel.from_pretrained(
                'cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
                subfolder='transformer',
                torch_dtype=torch.bfloat16,
                device_map={"": 0},  # Força tudo na GPU 0, evita CPU offload
                cache_dir='/root/.cache/huggingface',
                low_cpu_mem_usage=True
            ),
            transformer_2=WanTransformer3DModel.from_pretrained(
                'cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
                subfolder='transformer_2',
                torch_dtype=torch.bfloat16,
                device_map={"": 0},  # Força tudo na GPU 0, evita CPU offload
                cache_dir='/root/.cache/huggingface',
                low_cpu_mem_usage=True
            ),
            torch_dtype=torch.bfloat16,
            cache_dir='/root/.cache/huggingface'
        )
        # Garantir que o pipeline está na GPU
        pipe = pipe.to("cuda")
    else:
        # Configuração para CPU (fallback)
        print("Carregando modelo para CPU...")
        pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            transformer=WanTransformer3DModel.from_pretrained(
                'cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
                subfolder='transformer',
                torch_dtype=torch.float32,
                device_map="cpu",
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=True,
                offload_folder=offload_dir
            ),
            transformer_2=WanTransformer3DModel.from_pretrained(
                'cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
                subfolder='transformer_2',
                torch_dtype=torch.float32,
                device_map="cpu",
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=True,
                offload_folder=offload_dir
            ),
            torch_dtype=torch.float32,
            cache_dir=CACHE_DIR
        )
        # Garantir que o pipeline está na CPU
        pipe = pipe.to("cpu")

    print(f"Modelo carregado com sucesso no dispositivo: {device}")

except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    raise

for i in range(3):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

# Otimização mais conservadora
try:
    optimize_pipeline_(pipe,
                       # Tamanho menor para otimização
                       image=Image.new('RGB', (512, 512)),
                       prompt='test prompt',
                       height=512,
                       width=512,
                       num_frames=8,  # Menos frames para otimização
                       )
except Exception as e:
    print(f"Otimização falhou, mas continuando: {e}")


def get_duration(
    input_image,
    prompt,
    steps,
    negative_prompt,
    duration_seconds,
    guidance_scale,
    guidance_scale_2,
    seed,
    randomize_seed,
    progress,
):
    return int(steps) * 15


@spaces.GPU(duration=get_duration)
def generate_video(
    input_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate video and save to persistent storage"""

    if input_image is None:
        raise gr.Error("Please upload an input image.")

    num_frames = np.clip(int(round(duration_seconds * FIXED_FPS)),
                         MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    # Detectar o device real do modelo pipeline
    model_device = next(pipe.transformer.parameters()).device
    print(f"Modelo está no device: {model_device}")

    # Criar generator no mesmo device que o modelo
    generator = torch.Generator(device=model_device).manual_seed(current_seed)

    # Usar diretório temporário no disco local para processamento
    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as temp_dir:
        output_frames_list = pipe(
            image=resized_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=resized_image.height,
            width=resized_image.width,
            num_frames=num_frames,
            guidance_scale=float(guidance_scale),
            guidance_scale_2=float(guidance_scale_2),
            num_inference_steps=int(steps),
            generator=generator,
        ).frames[0]

        # Salvar no diretório persistente com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"wan_video_{timestamp}_seed_{current_seed}.mp4"
        video_path = os.path.join(OUTPUT_DIR, video_filename)

        print(f"Salvando vídeo em: {video_path}")

        export_to_video(output_frames_list, video_path, fps=FIXED_FPS)

        # Salvar metadados do vídeo
        metadata_path = os.path.join(OUTPUT_DIR, f"metadata_{timestamp}.txt")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Negative Prompt: {negative_prompt}\n")
            f.write(f"Steps: {steps}\n")
            f.write(f"Duration: {duration_seconds}s\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(f"Guidance Scale: {guidance_scale}\n")
            f.write(f"Guidance Scale 2: {guidance_scale_2}\n")
            f.write(f"Seed: {current_seed}\n")
            f.write(
                f"Image Size: {resized_image.width}x{resized_image.height}\n")
            f.write(f"Video File: {video_filename}\n")

    return video_path, current_seed


def list_generated_videos():
    """List all generated videos in the output directory"""
    try:
        videos = []
        if os.path.exists(OUTPUT_DIR):
            for file in os.listdir(OUTPUT_DIR):
                if file.endswith('.mp4'):
                    file_path = os.path.join(OUTPUT_DIR, file)
                    file_size = os.path.getsize(file_path)
                    videos.append((file, f"{file_size / 1024 / 1024:.1f} MB"))
        return videos
    except Exception as e:
        print(f"Erro ao listar vídeos: {e}")
        return []


def show_disk_usage():
    """Mostrar uso do disco"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(MODELS_DIR)

        return f"""
        **Uso do Disco E:\\Docker\\wan:**
        - Total: {total // (1024**3):.1f} GB
        - Usado: {used // (1024**3):.1f} GB  
        - Livre: {free // (1024**3):.1f} GB
        """
    except Exception as e:
        return f"Erro ao verificar uso do disco: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# Fast 4 steps Wan 2.2 I2V (14B) with Lightning LoRA")
    gr.Markdown("Todos os modelos, cache e vídeos salvos em E:\\Docker\\wan")

    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image")
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=3.5, label="Duration (seconds)",
                                               info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps.")

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt", value=default_negative_prompt, lines=3)
                seed_input = gr.Slider(
                    label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(
                    label="Randomize seed", value=True, interactive=True)
                steps_slider = gr.Slider(
                    minimum=1, maximum=30, step=1, value=6, label="Inference Steps")
                guidance_scale_input = gr.Slider(
                    minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale - high noise stage")
                guidance_scale_2_input = gr.Slider(
                    minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2 - low noise stage")

            generate_button = gr.Button("Generate Video", variant="primary")

            with gr.Accordion("Generated Videos & Disk Usage", open=False):
                refresh_button = gr.Button("Refresh List", variant="secondary")
                disk_usage = gr.Markdown()
                video_list = gr.Dataframe(
                    headers=["Filename", "Size"],
                    datatype=["str", "str"],
                    label="Saved Videos",
                    interactive=False
                )
                gr.Markdown(f"**Paths:**")
                gr.Markdown(f"- Models & Cache: `E:\\Docker\\wan`")
                gr.Markdown(f"- Videos: `E:\\Docker\\wan\\outputs`")

        with gr.Column():
            video_output = gr.Video(
                label="Generated Video", autoplay=True, interactive=False)

    def refresh_all():
        return list_generated_videos(), show_disk_usage()

    ui_inputs = [
        input_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input, seed_input, randomize_seed_checkbox
    ]

    generate_button.click(fn=generate_video, inputs=ui_inputs, outputs=[
                          video_output, seed_input])
    refresh_button.click(fn=refresh_all, outputs=[video_list, disk_usage])
    demo.load(fn=refresh_all, outputs=[video_list, disk_usage])

    # Adicionar exemplos
    gr.Examples(
        examples=[
            [
                "wan_i2v_input.JPG",
                "POV selfie video, white cat with sunglasses standing on surfboard, relaxed smile, tropical beach behind (clear water, green hills, blue sky with clouds). Surfboard tips, cat falls into ocean, camera plunges underwater with bubbles and sunlight beams. Brief underwater view of cat's face, then cat resurfaces, still filming selfie, playful summer vacation mood.",
                4,
            ],
            [
                "wan22_input_2.jpg",
                "A sleek lunar vehicle glides into view from left to right, kicking up moon dust as astronauts in white spacesuits hop aboard with characteristic lunar bouncing movements. In the distant background, a VTOL craft descends straight down and lands silently on the surface. Throughout the entire scene, ethereal aurora borealis ribbons dance across the star-filled sky, casting shimmering curtains of green, blue, and purple light that bathe the lunar landscape in an otherworldly, magical glow.",
                4,
            ],
            [
                "kill_bill.jpeg",
                "Uma Thurman's character, Beatrix Kiddo, holds her razor-sharp katana blade steady in the cinematic lighting. Suddenly, the polished steel begins to soften and distort, like heated metal starting to lose its structural integrity. The blade's perfect edge slowly warps and droops, molten steel beginning to flow downward in silvery rivulets while maintaining its metallic sheen.",
                6,
            ],
        ],
        inputs=[input_image_component, prompt_input, steps_slider],
        outputs=[video_output, seed_input],
        fn=generate_video,
        cache_examples="lazy"
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
