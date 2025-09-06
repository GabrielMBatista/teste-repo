import torch
import psutil
import os


class RTX3060Optimizer:
    def __init__(self):
        self.gpu_memory = 12.0  # GB
        self.target_utilization = 0.70  # 70% target

    def optimize_settings(self):
        """Otimiza configurações para RTX 3060"""
        if torch.cuda.is_available():
            # Configurações específicas para RTX 3060
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Otimizações de memória para 12GB
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,garbage_collection_threshold:0.6,expandable_segments:True'

            print("✅ RTX 3060 optimizations applied")
            return True
        return False

    def get_optimal_batch_size(self, base_resolution):
        """Calcula batch size ótimo baseado na resolução"""
        memory_per_pixel = 0.0001  # Estimativa
        available_memory = self.gpu_memory * 0.8  # 80% da memória

        pixels = base_resolution[0] * base_resolution[1]
        optimal_batch = int(available_memory / (pixels * memory_per_pixel))

        return max(1, min(optimal_batch, 4))

    def get_optimal_resolution(self, target_quality="high"):
        """Retorna resolução ótima baseada na qualidade desejada"""
        resolutions = {
            "low": (384, 512),
            "medium": (480, 640),
            "high": (512, 768),
            "ultra": (640, 832)
        }
        return resolutions.get(target_quality, resolutions["high"])

    def monitor_usage(self):
        """Monitora uso atual da GPU"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            utilization = allocated / self.gpu_memory

            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'utilization_percent': utilization * 100,
                'target_utilization': self.target_utilization * 100,
                'can_increase_load': utilization < self.target_utilization
            }
        return None


# Instância global
rtx3060_optimizer = RTX3060Optimizer()
