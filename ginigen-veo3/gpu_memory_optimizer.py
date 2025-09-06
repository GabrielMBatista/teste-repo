import torch
import gc
import psutil
import os


class GPUMemoryOptimizer:
    def __init__(self, target_gpu_usage=0.85, target_ram_usage=0.70):
        self.target_gpu_usage = target_gpu_usage  # 85% GPU
        self.target_ram_usage = target_ram_usage   # 70% RAM max

    def optimize_for_gpu_priority(self):
        """Configura sistema para priorizar GPU sobre RAM"""

        # Configurações PyTorch para mais GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False

        # Configurações de ambiente
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048,garbage_collection_threshold:0.8,expandable_segments:True'
        os.environ['PYTORCH_JIT'] = '1'
        os.environ['PYTORCH_TENSOREXPR_FALLBACK'] = '0'
        os.environ['CUDA_CACHE_DISABLE'] = '0'

        print("✅ Configured for GPU priority")

    def get_memory_status(self):
        """Retorna status detalhado de memória"""
        # GPU Memory
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_usage = gpu_allocated / gpu_total

        # RAM Memory
        ram = psutil.virtual_memory()
        ram_usage = ram.percent / 100
        ram_available_gb = ram.available / 1024**3

        return {
            'gpu_allocated_gb': gpu_allocated,
            'gpu_total_gb': gpu_total,
            'gpu_usage_percent': gpu_usage * 100,
            'gpu_target_percent': self.target_gpu_usage * 100,
            'ram_usage_percent': ram_usage * 100,
            'ram_available_gb': ram_available_gb,
            'ram_target_percent': self.target_ram_usage * 100,
            'is_gpu_underutilized': gpu_usage < self.target_gpu_usage,
            'is_ram_overused': ram_usage > self.target_ram_usage
        }

    def suggest_optimizations(self):
        """Sugere otimizações baseadas no uso atual"""
        status = self.get_memory_status()
        suggestions = []

        if status['is_gpu_underutilized']:
            suggestions.append(
                "🚀 GPU underutilized - increase batch size or resolution")
            suggestions.append("📈 Consider using higher precision (bfloat16)")
            suggestions.append("🎯 Enable more parallel processing")

        if status['is_ram_overused']:
            suggestions.append("⚠️ RAM overused - move more processing to GPU")
            suggestions.append("🔄 Disable CPU offloading")
            suggestions.append("💾 Keep models on GPU between generations")

        return suggestions

    def minimal_cleanup(self):
        """Limpeza mínima que não move dados da GPU"""
        torch.cuda.empty_cache()
        # Não fazer gc.collect() que pode mover dados

    def print_status(self):
        """Imprime status detalhado e visual de memória"""
        status = self.get_memory_status()

        print("┌" + "─" * 48 + "┐")
        print("│" + " " * 15 + "GPU MEMORY STATUS" + " " * 15 + "│")
        print("├" + "─" * 48 + "┤")

        # Barra visual da GPU
        gpu_bars = int(status['gpu_usage_percent'] / 5)  # 20 chars = 100%
        gpu_empty = 20 - gpu_bars
        gpu_bar = "█" * gpu_bars + "░" * gpu_empty

        print(f"│ GPU: [{gpu_bar}] {status['gpu_usage_percent']:.1f}% │")
        print(f"│ Used: {status['gpu_allocated_gb']:.2f}GB / {status['gpu_total_gb']:.2f}GB" + " " * (
            48 - 25 - len(f"{status['gpu_allocated_gb']:.2f}GB / {status['gpu_total_gb']:.2f}GB")) + "│")

        # Status da RAM
        ram_bars = int(status['ram_usage_percent'] / 5)
        ram_empty = 20 - ram_bars
        ram_bar = "█" * ram_bars + "░" * ram_empty

        print("├" + "─" * 48 + "┤")
        print(f"│ RAM: [{ram_bar}] {status['ram_usage_percent']:.1f}% │")
        print(f"│ Available: {status['ram_available_gb']:.1f}GB" + " " *
              (48 - 15 - len(f"{status['ram_available_gb']:.1f}GB")) + "│")

        # Sugestões
        suggestions = self.suggest_optimizations()
        if suggestions:
            print("├" + "─" * 48 + "┤")
            print("│" + " " * 18 + "SUGGESTIONS" + " " * 18 + "│")
            print("├" + "─" * 48 + "┤")
            for suggestion in suggestions[:2]:  # Mostrar apenas 2 sugestões
                suggestion_short = suggestion[:46]
                print(f"│ {suggestion_short}" + " " *
                      (48 - 2 - len(suggestion_short)) + "│")

        print("└" + "─" * 48 + "┘")

    def get_loading_progress_bar(self, current_step, total_steps, step_name):
        """Retorna uma barra de progresso visual para carregamento"""
        progress = current_step / total_steps
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        percentage = progress * 100

        return f"[{bar}] {percentage:.1f}% - {step_name}"


# Instância global
gpu_optimizer = GPUMemoryOptimizer()
