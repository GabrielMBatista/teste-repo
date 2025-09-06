from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate as mmaudio_generate,
                                load_video, make_video, setup_eval_logging)
from transformers import utils as transformers_utils
import huggingface_hub
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.flow_matching import FlowMatching
from src.transformer_wan_nag import NagWanTransformer3DModel
from src.pipeline_wan_nag import NAGWanPipeline
import torch.nn.functional as F
import traceback
from huggingface_hub import hf_hub_download
import tempfile
import gradio as gr
from diffusers import AutoModel
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
import torchaudio
import numpy as np
import torch
import types
import random
import spaces
import logging
import os
from pathlib import Path
from datetime import datetime
import re
import gc
from gpu_memory_optimizer import GPUMemoryOptimizer

# Configurações de memória CUDA ANTES de importar torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:3072,garbage_collection_threshold:0.9,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Patch for scaled_dot_product_attention to fix enable_gqa issue
original_sdpa = F.scaled_dot_product_attention


def patched_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=None):
    kwargs = {}
    if attn_mask is not None:
        kwargs['attn_mask'] = attn_mask
    if dropout_p != 0.0:
        kwargs['dropout_p'] = dropout_p
    if is_causal:
        kwargs['is_causal'] = is_causal
    if scale is not None:
        kwargs['scale'] = scale
    return original_sdpa(query, key, value, **kwargs)


F.scaled_dot_product_attention = patched_scaled_dot_product_attention

# MMAudio imports
try:
    import mmaudio
except ImportError:
    os.system("pip install -e .")
    import mmaudio

# Configuração do diretório de cache persistente - ALTERADO PARA /tmp/cache
CACHE_DIR = os.environ.get('CACHE_DIR', '/tmp/cache')
HF_CACHE = os.environ.get('HF_HOME', os.path.join(CACHE_DIR, 'huggingface'))
TORCH_CACHE = os.environ.get('TORCH_HOME', os.path.join(CACHE_DIR, 'torch'))
TRANSFORMERS_CACHE_DIR = os.environ.get(
    'TRANSFORMERS_CACHE', os.path.join(CACHE_DIR, 'transformers'))

# Garantir que os diretórios existam
os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(TORCH_CACHE, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE_DIR, exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, 'mmaudio'), exist_ok=True)
os.makedirs('/tmp/models', exist_ok=True)

# Configurar cache para huggingface - USANDO CAMINHOS ALTERADOS
os.environ['HF_HOME'] = HF_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR
os.environ['HF_HUB_CACHE'] = HF_CACHE
os.environ['HUGGINGFACE_HUB_CACHE'] = HF_CACHE
os.environ['HF_DATASETS_CACHE'] = HF_CACHE
os.environ['XDG_CACHE_HOME'] = CACHE_DIR

print(f"Cache directories configured:")
print(f"  HF_HOME: {HF_CACHE}")
print(f"  TRANSFORMERS_CACHE: {TRANSFORMERS_CACHE_DIR}")
print(f"  TORCH_HOME: {TORCH_CACHE}")
print(f"  CACHE_DIR: {CACHE_DIR}")

# NAG Video Settings - MOVER AQUI PARA DEPOIS DA CONFIGURAÇÃO DE CACHE
MOD_VALUE = 32
DEFAULT_DURATION_SECONDS = 2  # Padrão menor
DEFAULT_STEPS = 2         # Reduzido drasticamente para velocidade
DEFAULT_SEED = 2025
DEFAULT_H_SLIDER_VALUE = 480
DEFAULT_W_SLIDER_VALUE = 832
NEW_FORMULA_MAX_AREA = 480.0 * 832.0

SLIDER_MIN_H, SLIDER_MAX_H = 128, 896
SLIDER_MIN_W, SLIDER_MAX_W = 128, 896
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 129

DEFAULT_NAG_NEGATIVE_PROMPT = "Static, motionless, still, ugly, bad quality, worst quality, poorly drawn, low resolution, blurry, lack of details"
DEFAULT_AUDIO_NEGATIVE_PROMPT = "music, speech, voice, singing, narration"

# NAG Model Settings - DEPOIS DA CONFIGURAÇÃO DE CACHE
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
SUB_MODEL_ID = "vrgamedevgirl84/Wan14BT2VFusioniX"
SUB_MODEL_FILENAME = "Wan14BT2VFusioniX_fp16_.safetensors"
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

# MMAudio Settings - CORRIGIDO para usar cache
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
log = logging.getLogger()
device = 'cuda'
dtype = torch.bfloat16

# Configurar cache para MMAudio ANTES de criar o config
os.environ['TORCH_HOME'] = os.path.join(CACHE_DIR, 'torch')
os.makedirs(os.path.join(CACHE_DIR, 'torch'), exist_ok=True)

# PATCH para forçar downloads no cache correto
original_hf_hub_download = huggingface_hub.hf_hub_download


def patched_hf_hub_download(*args, **kwargs):
    """Força downloads para o cache correto"""
    if 'cache_dir' not in kwargs:
        kwargs['cache_dir'] = HF_CACHE
    print(f"Downloading to HF cache: {kwargs.get('cache_dir')}")
    return original_hf_hub_download(*args, **kwargs)


huggingface_hub.hf_hub_download = patched_hf_hub_download

# PATCH para transformers
if hasattr(transformers_utils, 'TRANSFORMERS_CACHE'):
    transformers_utils.TRANSFORMERS_CACHE = TRANSFORMERS_CACHE_DIR

audio_model_config: ModelConfig = all_model_cfg['large_44k_v2']

# Redirecionar paths do modelo para o cache
original_download = audio_model_config.download_if_needed


def download_to_cache():
    """Download models to our cache directory"""
    print(f"Downloading MMAudio models to cache: {CACHE_DIR}")

    # Verificar se já existe no cache
    mmaudio_cache = os.path.join(CACHE_DIR, 'mmaudio')
    # Garantir que o diretório exista
    os.makedirs(mmaudio_cache, exist_ok=True)

    result = original_download()

    # Copiar model_path para cache se não estiver lá
    if hasattr(audio_model_config, 'model_path') and audio_model_config.model_path:
        cached_model_path = os.path.join(
            mmaudio_cache, os.path.basename(audio_model_config.model_path))
        if not os.path.exists(cached_model_path) and os.path.exists(audio_model_config.model_path):
            import shutil
            print(f"Copying model to cache: {cached_model_path}")
            shutil.copy2(audio_model_config.model_path, cached_model_path)
            audio_model_config.model_path = cached_model_path
        elif os.path.exists(cached_model_path):
            print(f"Using existing cached model: {cached_model_path}")
            audio_model_config.model_path = cached_model_path

    return result


audio_model_config.download_if_needed = download_to_cache
audio_model_config.download_if_needed()
setup_eval_logging()

# Configurações agressivas de memória
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Limite de memória GPU (8GB)
GPU_MEMORY_LIMIT = 8.0  # GB
MAX_PIXELS = 384 * 256   # Resolução ainda mais conservadora para velocidade
MAX_FRAMES = 32          # Frames máximos reduzidos drasticamente

# Configurações otimizadas para RTX 3060 (12GB) - FULL GPU MODE
GPU_MEMORY_LIMIT = 12.0  # GB - RTX 3060 tem 12GB
MAX_PIXELS = 640 * 480   # Resolução melhor para 12GB
MAX_FRAMES = 64          # Mais frames para 12GB
DEFAULT_DURATION_SECONDS = 3  # Duração maior
DEFAULT_STEPS = 4         # Mais steps para melhor qualidade

# Configurações agressivas para GPU (menos RAM)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Instanciar o otimizador de memória
gpu_optimizer = GPUMemoryOptimizer()

# Função para limpeza de memória


def clear_cuda_cache():
    """Limpeza otimizada para manter mais na GPU"""
    if torch.cuda.is_available():
        # Menos limpeza agressiva para manter dados na GPU
        torch.cuda.empty_cache()
        print("✅ GPU cache optimized (minimal cleanup)")


def get_gpu_memory():
    """Retorna informações de memória GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB, Total: {total:.2f}GB"
    return "CUDA not available"


def check_memory_available(required_gb=2.0):
    """Verifica se há memória suficiente disponível"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        available = total - allocated
        return available >= required_gb
    return False


# Variáveis globais para carregamento sequencial
video_pipeline = None
audio_models = None

# Função para carregar apenas pipeline de vídeo


def load_video_pipeline():
    """Carrega pipeline otimizado para GPU com carregamento gradual"""
    global video_pipeline

    if video_pipeline is not None:
        return video_pipeline

    try:
        print("=" * 60)
        print("🚀 LOADING VIDEO PIPELINE (RTX 3060 CONSERVATIVE)")
        print("=" * 60)

        print(f"📊 Initial state: {get_gpu_memory()}")
        gpu_optimizer.print_status()
        clear_cuda_cache()

        # STEP 1: VAE (pequeno, sem problemas)
        print("\n🔄 STEP 1/6: Loading VAE...")
        print(f"📍 Cache directory: {HF_CACHE}")
        vae = AutoencoderKLWan.from_pretrained(
            MODEL_ID,
            subfolder="vae",
            torch_dtype=torch.float16,
            cache_dir=HF_CACHE,
            local_files_only=False,
            low_cpu_mem_usage=True
        )
        vae = vae.to("cuda")
        print(f"✅ VAE loaded successfully: {get_gpu_memory()}")

        # STEP 2: Download transformer (sem carregar na GPU ainda)
        print("\n🔄 STEP 2/6: Downloading Transformer...")
        print(f"📍 Repository: {SUB_MODEL_ID}")
        print(f"📍 File: {SUB_MODEL_FILENAME}")
        wan_path = hf_hub_download(
            repo_id=SUB_MODEL_ID,
            filename=SUB_MODEL_FILENAME,
            cache_dir=HF_CACHE,
            local_files_only=False
        )
        print(f"✅ Transformer downloaded to: {wan_path}")

        # STEP 3: Carregar transformer APENAS na CPU primeiro
        print("\n🔄 STEP 3/6: Loading Transformer to CPU...")
        print(f"📍 Loading from: {wan_path}")
        print("⚠️ Loading large 14B model - this may take time...")

        # Carregar na CPU primeiro para economizar VRAM
        transformer = NagWanTransformer3DModel.from_single_file(
            wan_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # NÃO mover para GPU ainda
        )
        print(f"✅ Transformer loaded to CPU: {get_gpu_memory()}")

        # STEP 4: Criar pipeline na CPU
        print("\n🔄 STEP 4/6: Creating Pipeline on CPU...")
        print(f"📍 Using cache: {HF_CACHE}")

        # Mover VAE de volta para CPU temporariamente
        vae = vae.to("cpu")
        clear_cuda_cache()
        print(f"📍 VAE moved to CPU, GPU cleared: {get_gpu_memory()}")

        pipeline = NAGWanPipeline.from_pretrained(
            MODEL_ID,
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.float16,
            cache_dir=HF_CACHE,
            local_files_only=False,
            low_cpu_mem_usage=True
            # Manter tudo na CPU por enquanto
        )

        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config,
            flow_shift=3.0
        )
        print(f"✅ Pipeline created on CPU: {get_gpu_memory()}")

        # STEP 5: Mover componentes GRADUALMENTE para GPU
        print("\n🔄 STEP 5/6: Moving components to GPU gradually...")

        # Primeiro o VAE (pequeno)
        print("🔄 Moving VAE to GPU...")
        pipeline.vae = pipeline.vae.to("cuda")
        print(f"✅ VAE on GPU: {get_gpu_memory()}")

        # Limpeza antes do transformer
        clear_cuda_cache()

        # Depois o transformer (grande) - pode falhar aqui
        print("🔄 Moving Transformer to GPU...")
        try:
            pipeline.transformer = pipeline.transformer.to("cuda")
            print(f"✅ Transformer on GPU: {get_gpu_memory()}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("❌ Transformer too large for GPU - using CPU offload")
                # Usar CPU offload apenas para o transformer
                from accelerate import cpu_offload
                pipeline.transformer = cpu_offload(
                    pipeline.transformer, execution_device=0)
                print(f"✅ Transformer using CPU offload: {get_gpu_memory()}")
            else:
                raise e

        # Text encoder e outros componentes menores
        print("🔄 Moving other components to GPU...")
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            pipeline.text_encoder = pipeline.text_encoder.to("cuda")
        print(f"✅ Other components on GPU: {get_gpu_memory()}")

        # STEP 6: Configurações finais
        print("\n🔄 STEP 6/6: Final setup...")

        # Configurações necessárias
        pipeline.transformer.__class__.attn_processors = NagWanTransformer3DModel.attn_processors
        pipeline.transformer.__class__.set_attn_processor = NagWanTransformer3DModel.set_attn_processor
        pipeline.transformer.__class__.forward = NagWanTransformer3DModel.forward

        video_pipeline = pipeline

        print("\n🎉 PIPELINE LOADING COMPLETE!")
        print("=" * 60)
        print(f"📊 Final GPU status: {get_gpu_memory()}")
        gpu_optimizer.print_status()
        print("=" * 60)

        return pipeline

    except Exception as e:
        print(f"\n❌ ERROR in pipeline loading: {e}")
        print("=" * 60)
        traceback.print_exc()
        clear_cuda_cache()
        return None

# Função para carregar modelos de áudio


def load_audio_models():
    """Carrega modelos de áudio com status detalhado"""
    global audio_models

    if audio_models is not None:
        print("✅ Audio models already loaded")
        return audio_models

    try:
        print("\n" + "=" * 60)
        print("🎵 LOADING AUDIO MODELS")
        print("=" * 60)

        print(f"📊 Before audio loading: {get_gpu_memory()}")

        model_cache_dir = os.path.join(CACHE_DIR, 'mmaudio')
        os.makedirs(model_cache_dir, exist_ok=True)

        # STEP 1: Check cached model
        print("\n🔄 STEP 1/3: Checking cached models...")
        if hasattr(audio_model_config, 'model_path'):
            cached_model_path = os.path.join(
                model_cache_dir, os.path.basename(audio_model_config.model_path))
            if os.path.exists(cached_model_path):
                audio_model_config.model_path = cached_model_path
                print(f"✅ Using cached model: {cached_model_path}")
            else:
                print(f"📍 Model will be downloaded to: {cached_model_path}")

        seq_cfg = audio_model_config.seq_cfg

        # STEP 2: Load Audio Network
        print("\n🔄 STEP 2/3: Loading Audio Network...")
        print(f"📍 Model: {audio_model_config.model_name}")
        net: MMAudio = get_my_mmaudio(audio_model_config.model_name)

        print("🔄 Loading model weights...")
        weights = torch.load(audio_model_config.model_path,
                             map_location=device, weights_only=True)
        net.load_weights(weights)
        net = net.to(device, dtype).eval()
        print(f"✅ Audio network loaded: {get_gpu_memory()}")

        # STEP 3: Load Feature Utils
        print("\n🔄 STEP 3/3: Loading Feature Utils...")
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=audio_model_config.vae_path,
            synchformer_ckpt=audio_model_config.synchformer_ckpt,
            enable_conditions=True,
            mode=audio_model_config.mode,
            bigvgan_vocoder_ckpt=audio_model_config.bigvgan_16k_path,
            need_vae_encoder=False
        )
        feature_utils = feature_utils.to(device, dtype).eval()

        audio_models = (net, feature_utils, seq_cfg)

        print("\n🎉 AUDIO MODELS LOADING COMPLETE!")
        print("=" * 60)
        print(f"📊 Final GPU status: {get_gpu_memory()}")
        gpu_optimizer.print_status()
        print("=" * 60)

        return audio_models

    except Exception as e:
        print(f"\n❌ ERROR in audio models loading: {e}")
        print("=" * 60)
        traceback.print_exc()
        clear_cuda_cache()
        return None

# REMOVER funções de unload - manter tudo na GPU


def unload_video_pipeline():
    """DESABILITADO - manter pipeline na GPU"""
    print("🚀 Keeping video pipeline on GPU for performance")


def unload_audio_models():
    """DESABILITADO - manter modelos na GPU"""
    print("🚀 Keeping audio models on GPU for performance")

# 비디오 프롬프트를 오디오 프롬프트로 변환하는 함수


def extract_audio_description(video_prompt):
    """비디오 프롬프트에서 오디오 관련 설명 추출/변환"""
    # 키워드 매핑
    audio_keywords = {
        'car': 'car engine sound, vehicle noise',
        'porsche': 'sports car engine roar, exhaust sound',
        'guitar': 'electric guitar playing, guitar music',
        'concert': 'crowd cheering, live music, applause',
        'motorcycle': 'motorcycle engine sound, motor rumble',
        'highway': 'traffic noise, road ambience',
        'rain': 'rain sounds, water drops',
        'wind': 'wind blowing sound',
        'ocean': 'ocean waves, water sounds',
        'city': 'urban ambience, city traffic sounds',
        'singer': 'singing voice, vocals',
        'crowd': 'crowd noise, people talking',
        'flames': 'fire crackling sound',
        'pyro': 'fire whoosh, flame burst sound',
        'explosion': 'explosion sound, blast',
        'countryside': 'nature ambience, birds chirping',
        'wheat fields': 'wind through grass, rural ambience',
        'engine': 'motor sound, mechanical noise',
        'flat-six engine': 'sports car engine sound',
        'roaring': 'loud engine roar',
        'thunderous': 'loud booming sound',
        'child': 'children playing sounds',
        'running': 'footsteps sound',
        'woman': 'ambient sounds',
        'phone': 'subtle electronic ambience',
        'advertisement': 'modern ambient sounds'
    }

    # 간단한 키워드 기반 변환
    audio_descriptions = []
    lower_prompt = video_prompt.lower()

    for key, value in audio_keywords.items():
        if key in lower_prompt:
            audio_descriptions.append(value)

    # 기본값 설정
    if not audio_descriptions:
        # 프롬프트에 명시적인 오디오 설명이 있는지 확인
        if 'sound' in lower_prompt or 'audio' in lower_prompt or 'noise' in lower_prompt:
            # 프롬프트에서 오디오 관련 부분만 추출
            audio_pattern = r'([^.]*(?:sound|audio|noise|music|voice|roar|rumble)[^.]*)'
            matches = re.findall(audio_pattern, lower_prompt, re.IGNORECASE)
            if matches:
                return ', '.join(matches)

        # 기본 ambient sound
        return "ambient environmental sounds matching the scene"

    return ', '.join(audio_descriptions)

# Audio generation function com gerenciamento sequencial


@torch.inference_mode()
def add_audio_to_video(video_path, prompt, audio_custom_prompt, audio_negative_prompt, audio_steps, audio_cfg_strength, duration):
    """Generate and add audio to video using MMAudio (GPU optimized)"""
    try:
        print(f"Audio generation start (GPU optimized): {get_gpu_memory()}")

        # Carregar modelos de áudio (se não estiverem carregados)
        audio_models_tuple = load_audio_models()
        if audio_models_tuple is None:
            print("Audio models not available, returning video without audio")
            return video_path

        audio_net, audio_feature_utils, audio_seq_cfg = audio_models_tuple
        print("✅ Audio models already on GPU")

        # Processar prompt de áudio
        if audio_custom_prompt and audio_custom_prompt.strip():
            audio_prompt = audio_custom_prompt.strip()
        else:
            audio_prompt = extract_audio_description(prompt)

        print(f"Original prompt: {prompt}")
        print(f"Audio prompt: {audio_prompt}")

        rng = torch.Generator(device=device)
        rng.manual_seed(random.randint(0, 2**32 - 1))
        fm = FlowMatching(min_sigma=0, inference_mode='euler',
                          num_steps=audio_steps)

        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        # CORRIGIDO: duration_sec em vez de duration_secs
        duration = video_info.duration_sec
        clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)
        audio_seq_cfg.duration = duration
        audio_net.update_seq_lengths(
            audio_seq_cfg.latent_seq_len, audio_seq_cfg.clip_seq_len, audio_seq_cfg.sync_seq_len)

        enhanced_negative = f"{audio_negative_prompt}, distortion, static noise, silence, random beeps"

        audios = mmaudio_generate(clip_frames,
                                  sync_frames, [audio_prompt],
                                  negative_text=[enhanced_negative],
                                  feature_utils=audio_feature_utils,
                                  net=audio_net,
                                  fm=fm,
                                  rng=rng,
                                  cfg_strength=audio_cfg_strength)
        audio = audios.float().cpu()[0]

        # Create video with audio
        video_with_audio_path = tempfile.NamedTemporaryFile(
            delete=False, suffix='.mp4').name
        make_video(video_info, video_with_audio_path, audio,
                   sampling_rate=audio_seq_cfg.sampling_rate)

        print(f"Audio generation complete (kept on GPU): {get_gpu_memory()}")
        return video_with_audio_path

    except Exception as e:
        print(f"Error in audio generation: {e}")
        clear_cuda_cache()
        return video_path

# Combined generation function com carregamento sequencial


@spaces.GPU(duration=lambda *args: 180)  # Mais tempo para qualidade
def generate_video_with_audio(
        prompt,
        nag_negative_prompt, nag_scale,
        height=DEFAULT_H_SLIDER_VALUE, width=DEFAULT_W_SLIDER_VALUE, duration_seconds=DEFAULT_DURATION_SECONDS,
        steps=DEFAULT_STEPS,
        seed=DEFAULT_SEED, randomize_seed=False,
        enable_audio=True, audio_custom_prompt="",
        audio_negative_prompt=DEFAULT_AUDIO_NEGATIVE_PROMPT,
        audio_steps=25, audio_cfg_strength=4.0,  # Valores melhores para qualidade
):
    try:
        print(f"RTX 3060 Generation start: {get_gpu_memory()}")
        clear_cuda_cache()

        # Carregar pipeline de vídeo
        pipe = load_video_pipeline()
        if pipe is None:
            return None, DEFAULT_SEED

        # Dimensões otimizadas para RTX 3060 (12GB)
        target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
        target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)

        # Limite para RTX 3060 - mais generoso
        if target_h * target_w > MAX_PIXELS:
            ratio = (MAX_PIXELS / (target_h * target_w)) ** 0.5
            target_h = max(MOD_VALUE, int(target_h * ratio) //
                           MOD_VALUE * MOD_VALUE)
            target_w = max(MOD_VALUE, int(target_w * ratio) //
                           MOD_VALUE * MOD_VALUE)
            print(f"Resolution adjusted to {target_w}x{target_h} for RTX 3060")

        # Frames otimizados para 12GB
        duration_seconds = min(
            4, max(1, int(duration_seconds)))  # Máximo 4 segundos
        # 12 FPS para qualidade
        num_frames = min(48, max(16, int(duration_seconds * 12)))
        steps = min(6, max(2, int(steps)))  # Até 6 steps para qualidade

        current_seed = random.randint(
            0, MAX_SEED) if randomize_seed else int(seed)

        print(
            f"RTX 3060 OPTIMIZED: {target_w}x{target_h}, {num_frames} frames, {steps} steps")
        print(f"Before video generation: {get_gpu_memory()}")

        # Usar GPU de forma mais agressiva para RTX 3060
        pipe.to("cuda")
        with torch.inference_mode():
            # Usar bfloat16 para melhor qualidade na RTX 3060
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nag_output_frames_list = pipe(
                    prompt=prompt,
                    nag_negative_prompt=nag_negative_prompt,
                    # Escala maior para qualidade
                    nag_scale=min(12.0, nag_scale),
                    nag_tau=3.5,  # Valor original
                    nag_alpha=0.5,  # Valor original
                    height=target_h, width=target_w, num_frames=num_frames,
                    guidance_scale=0.,
                    num_inference_steps=steps,
                    generator=torch.Generator(
                        device="cuda").manual_seed(current_seed)
                ).frames[0]

        print(f"After video generation: {get_gpu_memory()}")

        # Salvar vídeo com FPS melhor
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            temp_video_path = tmpfile.name
        export_to_video(nag_output_frames_list,
                        temp_video_path, fps=12)  # 12 FPS

        # Liberar memória
        del nag_output_frames_list
        clear_cuda_cache()

        # Áudio com qualidade melhor
        if enable_audio:
            try:
                print("Adding high-quality audio...")
                final_video_path = add_audio_to_video(
                    temp_video_path, prompt, audio_custom_prompt,
                    audio_negative_prompt, min(25, audio_steps), min(
                        4.0, audio_cfg_strength), duration_seconds
                )
                if os.path.exists(temp_video_path) and final_video_path != temp_video_path:
                    os.remove(temp_video_path)
            except Exception as e:
                print(f"Audio generation failed: {e}")
                final_video_path = temp_video_path
        else:
            final_video_path = temp_video_path

        clear_cuda_cache()
        print(f"Generation complete: {get_gpu_memory()}")

        return final_video_path, current_seed

    except Exception as e:
        print(f"Error in video generation: {e}")
        clear_cuda_cache()
        return None, current_seed


# Configurações dinâmicas baseadas na escolha do usuário
QUALITY_PRESETS = {
    "speed": {
        "max_resolution": (320, 384),  # MUITO conservador para 14B
        "max_frames": 8,               # Mínimo absoluto
        "max_duration": 1,             # Apenas 1 segundo
        "max_steps": 2,                # Mínimo steps
        "fps": 8,
        "audio_steps": 5,              # Mínimo áudio
        "dtype": torch.float16,
        "description": "Ultra-fast for 14B model (~1-2 minutes)"
    },
    "balanced": {
        "max_resolution": (384, 448),  # Reduzido drasticamente
        "max_frames": 12,              # Muito reduzido
        "max_duration": 2,
        "max_steps": 3,
        "fps": 8,
        "audio_steps": 10,
        "dtype": torch.float16,
        "description": "Conservative for 14B model (~3-5 minutes)"
    },
    "quality": {
        "max_resolution": (448, 512),  # Máximo seguro para 14B
        "max_frames": 16,              # Reduzido para 14B
        "max_duration": 2,             # Máximo 2 segundos
        "max_steps": 4,                # Reduzido
        "fps": 12,
        "audio_steps": 15,
        "dtype": torch.float16,        # Manter float16
        "description": "Best quality for 14B model (~5-8 minutes)"
    }
}


def update_ui_based_on_preset(preset_name):
    """Atualiza os valores da UI baseado no preset selecionado"""
    preset = QUALITY_PRESETS[preset_name]
    max_h, max_w = preset["max_resolution"]

    return [
        gr.update(maximum=preset["max_duration"],
                  value=min(3, preset["max_duration"])),
        gr.update(maximum=preset["max_steps"], value=preset["max_steps"]),
        gr.update(maximum=max_h, value=min(480, max_h)),
        gr.update(maximum=max_w, value=min(640, max_w)),
        gr.update(value=preset["fps"]),
        gr.update(value=preset["audio_steps"]),
        gr.update(visible=True)
    ]


@spaces.GPU(duration=lambda *args: 300)
def generate_video_with_audio(
        prompt,
        nag_negative_prompt, nag_scale,
        quality_preset,
        height, width, duration_seconds,
        steps, fps_target,
        seed, randomize_seed,
        enable_audio, audio_custom_prompt,
        audio_negative_prompt, audio_steps, audio_cfg_strength,
        precision_mode
):
    try:
        preset = QUALITY_PRESETS[quality_preset]

        print("\n" + "🎬" * 20)
        print(f"🎬 STARTING VIDEO GENERATION (14B MODEL)")
        print(f"🎯 Preset: {quality_preset.upper()}")
        print(f"📊 Initial GPU: {get_gpu_memory()}")
        print("🎬" * 20)

        # Carregar pipeline de vídeo
        pipe = load_video_pipeline()
        if pipe is None:
            return None, DEFAULT_SEED

        print(f"\n✅ Pipeline ready (with CPU offload if needed)")

        # Parâmetros MUITO CONSERVADORES para 14B
        max_h, max_w = preset["max_resolution"]
        target_h = min(max_h, max(
            MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE))
        target_w = min(max_w, max(
            MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE))

        # Limites extremamente conservadores
        duration_seconds = min(
            preset["max_duration"], max(1, int(duration_seconds)))
        num_frames = min(preset["max_frames"], max(
            8, int(duration_seconds * fps_target)))
        steps = min(preset["max_steps"], max(1, int(steps)))
        current_seed = random.randint(
            0, MAX_SEED) if randomize_seed else int(seed)

        # FORÇAR limites ainda menores se necessário
        if target_h * target_w > preset["max_resolution"][0] * preset["max_resolution"][1]:
            target_h = preset["max_resolution"][0]
            target_w = preset["max_resolution"][1]
            print(
                f"⚠️ Resolution forced to preset limit: {target_w}x{target_h}")

        print(f"\n📋 GENERATION PARAMETERS (14B OPTIMIZED):")
        print(f"   Resolution: {target_w}x{target_h}")
        print(f"   Frames: {num_frames}")
        print(f"   Steps: {steps}")
        print(f"   FPS: {fps_target}")
        print(f"   Duration: {duration_seconds}s")
        print(f"   Seed: {current_seed}")

        # SEMPRE usar float16 para economizar memória
        dtype = torch.float16
        print(f"   Using dtype: {dtype} (forced for 14B model)")

        print(f"\n🎬 Starting video generation...")
        print(f"📊 GPU before generation: {get_gpu_memory()}")

        # Limpeza preventiva
        clear_cuda_cache()

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=dtype, enabled=True):
                nag_output_frames_list = pipe(
                    prompt=prompt,
                    nag_negative_prompt=nag_negative_prompt,
                    nag_scale=min(12.0, nag_scale),  # Reduzido
                    nag_tau=2.5,  # Reduzido para economizar memória
                    nag_alpha=0.3,  # Reduzido
                    height=target_h, width=target_w, num_frames=num_frames,
                    guidance_scale=0.,
                    num_inference_steps=steps,
                    generator=torch.Generator(
                        device="cuda").manual_seed(current_seed)
                ).frames[0]

        print(f"✅ Video generation complete!")
        print(f"📊 GPU after generation: {get_gpu_memory()}")

        # Salvar vídeo
        print(f"\n💾 Saving video at {fps_target} FPS...")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            temp_video_path = tmpfile.name
        export_to_video(nag_output_frames_list,
                        temp_video_path, fps=fps_target)
        print(f"✅ Video saved to: {temp_video_path}")

        # Limpeza agressiva para liberar memória para áudio
        del nag_output_frames_list
        torch.cuda.empty_cache()
        print(f"📊 GPU after cleanup: {get_gpu_memory()}")

        # Áudio apenas se habilitado e com configurações mínimas
        if enable_audio:
            print(f"\n🎵 Starting minimal audio generation...")
            print(f"   Audio steps: {min(audio_steps, 10)}")  # Máximo 10 steps

            try:
                final_video_path = add_audio_to_video(
                    temp_video_path, prompt, audio_custom_prompt,
                    audio_negative_prompt, min(
                        audio_steps, 10), audio_cfg_strength, duration_seconds
                )
                if os.path.exists(temp_video_path) and final_video_path != temp_video_path:
                    os.remove(temp_video_path)
                print(f"✅ Audio generation complete!")
            except Exception as e:
                print(f"❌ Audio generation failed: {e}")
                final_video_path = temp_video_path
        else:
            print(f"\n⏭️ Skipping audio generation")
            final_video_path = temp_video_path

        print(f"\n🎉 GENERATION COMPLETE!")
        print(f"📊 Final GPU status: {get_gpu_memory()}")
        print("🎬" * 20)

        return final_video_path, current_seed

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ GPU OUT OF MEMORY WITH 14B MODEL!")
            print(f"💡 The 14B model is too large for RTX 3060")
            print(f"💡 Try: Use a smaller model or reduce all parameters")
            print(f"📊 GPU at error: {get_gpu_memory()}")
            torch.cuda.empty_cache()
        else:
            print(f"❌ Runtime error: {e}")
        return None, current_seed
    except Exception as e:
        print(f"❌ Generation error: {e}")
        torch.cuda.empty_cache()
        return None, current_seed


# Examples with audio descriptions
examples = [
    ["Midnight highway outside a neon-lit city. A black 1973 Porsche 911 Carrera RS speeds at 120 km/h. Inside, a stylish singer-guitarist sings while driving, vintage sunburst guitar on the passenger seat. Sodium streetlights streak over the hood; RGB panels shift magenta to blue on the driver. Camera: drone dive, Russian-arm low wheel shot, interior gimbal, FPV barrel roll, overhead spiral. Neo-noir palette, rain-slick asphalt reflections, roaring flat-six engine blended with live guitar.", DEFAULT_NAG_NEGATIVE_PROMPT, 11],
    ["Arena rock concert packed with  20 000 fans. A flamboyant lead guitarist in leather jacket and mirrored aviators shreds a cherry-red Flying V on a thrust stage. Pyro flames shoot up on every downbeat, CO₂ jets burst behind. Moving-head spotlights swirl teal and amber, follow-spots rim-light the guitarist's hair. Steadicam 360-orbit, crane shot rising over crowd, ultra-slow-motion pick attack at 1 000 fps. Film-grain teal-orange grade, thunderous crowd roar mixes with screaming guitar solo.", DEFAULT_NAG_NEGATIVE_PROMPT, 11],
    ["Golden-hour countryside road winding through rolling wheat fields. A man and woman ride a vintage café-racer motorcycle, hair and scarf fluttering in the warm breeze. Drone chase shot reveals endless patchwork farmland; low slider along rear wheel captures dust trail. Sun-flare back-lights the riders, lens blooms on highlights. Soft acoustic rock underscore; engine rumble mixed at –8 dB. Warm pastel color grade, gentle film-grain for nostalgic vibe.", DEFAULT_NAG_NEGATIVE_PROMPT, 11],
]

# CSS styling - Fixed for better layout
css = """
/* Right column - video output */
.video-output {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    width: 100% !important;
    height: auto !important;
    min-height: 400px;
}

/* Ensure video container is responsive */
.video-output video {
    width: 100% !important;
    height: auto !important;
    max-height: 600px;
    object-fit: contain;
    display: block;
}

/* Remove any overlay or background from video container */
.video-output > div {
    background: transparent !important;
    padding: 0 !important;
}

/* Remove gradio's default video player overlay */
.video-output .wrap {
    background: transparent !important;
}

/* Ensure no gray overlay on video controls */
.video-output video::-webkit-media-controls-enclosure {
    background: transparent;
}

/* Prompt container styling */
.prompt-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

.prompt-input {
    border-radius: 8px;
    border: 2px solid #e1e5e9;
    transition: border-color 0.3s ease;
}

.prompt-input:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* Settings panel styling */
.settings-panel {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

.slider-container {
    margin: 10px 0;
}

/* Audio settings styling */
.audio-settings {
    background: #fff7ed;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 2px solid #fed7aa;
}

/* Generate button styling */
.generate-btn {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 15px 30px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.generate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* Container styling */
.container {
    text-align: center;
    margin-bottom: 30px;
}

.main-title {
    font-size: 2.5rem;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 1.1rem;
    color: #6b7280;
    margin-bottom: 20px;
}
"""

# Gradio interface com controles flexíveis
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div class="container">
            <h1 class="main-title">🎬 VEO3 Free - GPU Optimized</h1>
            <p class="subtitle">Wan2.1-T2V-14B + NAG + Audio Generation (Full GPU Mode)</p>
        </div>
    """)

    gr.HTML("""
        <div class='container' style='display:flex; justify-content:center; gap:12px; margin-bottom: 20px;'>
            <a href="https://huggingface.co/spaces/openfree/Best-AI" target="_blank">
                <img src="https://img.shields.io/static/v1?label=OpenFree&message=BEST%20AI%20Services&color=%230000ff&labelColor=%23000080&logo=huggingface&logoColor=%23ffa500&style=for-the-badge" alt="OpenFree badge">
            </a>

            <a href="https://discord.gg/openfreeai" target="_blank">
                <img src="https://img.shields.io/static/v1?label=Discord&message=Openfree%20AI&color=%230000ff&labelColor=%23800080&logo=discord&logoColor=white&style=for-the-badge" alt="Discord badge">
            </a>
        </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            with gr.Group(elem_classes="prompt-container"):
                prompt = gr.Textbox(
                    label="✨ Video Prompt (also used for audio generation)",
                    placeholder="Describe your video scene in detail...",
                    lines=3,
                    elem_classes="prompt-input"
                )

                with gr.Accordion("🎨 Advanced Video Settings", open=False):
                    nag_negative_prompt = gr.Textbox(
                        label="Video Negative Prompt",
                        value=DEFAULT_NAG_NEGATIVE_PROMPT,
                        lines=2,
                    )
                    nag_scale = gr.Slider(
                        label="NAG Scale",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.25,
                        value=11.0,
                        info="Higher values = stronger guidance"
                    )

            with gr.Group(elem_classes="settings-panel"):
                gr.Markdown("### ⚙️ Quality & Speed Settings")
                # Preset selector
                quality_preset = gr.Radio(
                    choices=["speed", "balanced", "quality"],
                    value="balanced",
                    label="🎯 Quality Preset",
                    info="Choose your speed/quality preference"
                )
                # Precision mode
                precision_mode = gr.Radio(
                    choices=[("Fast (float16)", "fast"),
                             ("High Quality (bfloat16)", "high")],
                    value="fast",
                    label="🔬 Precision Mode",
                    info="Higher precision = better quality but slower"
                )

                with gr.Row():
                    duration_seconds_input = gr.Slider(
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=3,
                        label="📱 Duration (seconds)",
                        elem_classes="slider-container"
                    )
                    steps_slider = gr.Slider(
                        minimum=1,
                        maximum=6,
                        step=1,
                        value=4,
                        label="🔄 Inference Steps",
                        elem_classes="slider-container"
                    )

                with gr.Row():
                    height_input = gr.Slider(
                        minimum=256,
                        maximum=640,
                        step=MOD_VALUE,
                        value=480,
                        label=f"📐 Height (×{MOD_VALUE})",
                        elem_classes="slider-container"
                    )
                    width_input = gr.Slider(
                        minimum=384,
                        maximum=832,
                        step=MOD_VALUE,
                        value=640,
                        label=f"📐 Width (×{MOD_VALUE})",
                        elem_classes="slider-container"
                    )
                # FPS control
                fps_target = gr.Slider(
                    minimum=8,
                    maximum=24,
                    step=2,
                    value=12,
                    label="🎬 Target FPS",
                    info="Higher FPS = smoother motion but longer processing"
                )

                # Dynamic info box
                preset_info = gr.HTML("""
                    <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px; padding: 12px; margin: 10px 0;">
                        <p style="margin: 0; color: #1976d2;">
                            ⚖️ <strong>Balanced Mode:</strong> Good balance (~4-6 minutes)
                        </p>
                    </div>
                """)

                with gr.Row():
                    seed_input = gr.Slider(
                        label="🌱 Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=DEFAULT_SEED,
                        interactive=True
                    )
                    randomize_seed_checkbox = gr.Checkbox(
                        label="🎲 Random Seed",
                        value=True,
                        interactive=True
                    )

            with gr.Group(elem_classes="audio-settings"):
                gr.Markdown("### 🎵 Audio Generation Settings")
                enable_audio = gr.Checkbox(
                    label="🔊 Enable Audio Generation",
                    value=True,
                    interactive=True
                )

                with gr.Column(visible=True) as audio_settings_group:
                    audio_custom_prompt = gr.Textbox(
                        label="Custom Audio Prompt (Optional)",
                        placeholder="Custom audio description",
                        value="",
                    )
                    audio_negative_prompt = gr.Textbox(
                        label="Audio Negative Prompt",
                        value=DEFAULT_AUDIO_NEGATIVE_PROMPT,
                    )

                    with gr.Row():
                        audio_steps = gr.Slider(
                            minimum=5,
                            maximum=50,
                            step=5,
                            value=20,
                            label="🎚️ Audio Steps",
                            info="More steps = better audio quality"
                        )
                        audio_cfg_strength = gr.Slider(
                            minimum=1.0,
                            maximum=8.0,
                            step=0.5,
                            value=4.0,
                            label="🎛️ Audio Guidance",
                            info="Strength of prompt guidance"
                        )

                # Toggle audio settings visibility
                enable_audio.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[enable_audio],
                    outputs=[audio_settings_group]
                )

            # Update UI when preset changes
            quality_preset.change(
                fn=update_ui_based_on_preset,
                inputs=[quality_preset],
                outputs=[
                    duration_seconds_input,
                    steps_slider,
                    height_input,
                    width_input,
                    fps_target,
                    audio_steps,
                    preset_info
                ]
            )

            # Update info display
            def update_preset_info(preset):
                preset_data = QUALITY_PRESETS[preset]
                icon = "⚡" if preset == "speed" else "⚖️" if preset == "balanced" else "🚀"
                return f"""
                    <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px; padding: 12px; margin: 10px 0;">
                        <p style="margin: 0; color: #1976d2;">
                            {icon} <strong>{preset.title()} Mode:</strong> {preset_data["description"]}
                        </p>
                    </div>
                """

            quality_preset.change(
                fn=update_preset_info,
                inputs=[quality_preset],
                outputs=[preset_info]
            )

            generate_button = gr.Button(
                "🚀 Generate Video (Full GPU Mode)",
                variant="primary",
                elem_classes="generate-btn"
            )

        with gr.Column(scale=5):
            video_output = gr.Video(
                label="Generated Video with Audio",
                autoplay=True,
                interactive=False,
                elem_classes="video-output",
                height=600
            )

            gr.HTML("""
                <div style="text-align: center; margin-top: 20px; color: #6b7280;">
                    <p>🚀 <strong>GPU Mode:</strong> Models kept on GPU for maximum performance</p>
                    <p>💡 <strong>RTX 3060:</strong> 12GB VRAM fully utilized</p>
                    <p>🎧 <strong>Audio:</strong> Synchronized with video generation</p>
                </div>
            """)

    # Examples section moved outside of columns
    with gr.Row():
        gr.Markdown("### 🎯 Example Prompts")

    gr.Examples(
        examples=examples,
        inputs=[prompt, nag_negative_prompt, nag_scale],
        outputs=None,  # Don't connect outputs to avoid index issues
        cache_examples=False
    )

    # Connect UI elements with new parameters
    ui_inputs = [
        prompt,
        nag_negative_prompt, nag_scale,
        quality_preset,  # NEW
        height_input, width_input, duration_seconds_input,
        steps_slider, fps_target,  # NEW
        seed_input, randomize_seed_checkbox,
        enable_audio, audio_custom_prompt, audio_negative_prompt,
        audio_steps, audio_cfg_strength,
        precision_mode  # NEW
    ]

    generate_button.click(
        fn=generate_video_with_audio,
        inputs=ui_inputs,
        outputs=[video_output, seed_input],
    )

if __name__ == "__main__":
    demo.queue().launch()
