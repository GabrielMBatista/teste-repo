import torch
import psutil
import time


class GPUMonitor:
    def __init__(self, target_gpu_usage=85):
        self.target_gpu_usage = target_gpu_usage

    def check_status(self):
        """Verifica status atual GPU vs RAM"""
        if torch.cuda.is_available():
            # GPU Status
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_total = torch.cuda.get_device_properties(
                0).total_memory / 1024**3
            gpu_percent = (gpu_allocated / gpu_total) * 100

            # RAM Status
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            ram_available = ram.available / 1024**3

            status = {
                'gpu_allocated_gb': gpu_allocated,
                'gpu_total_gb': gpu_total,
                'gpu_percent': gpu_percent,
                'gpu_target': self.target_gpu_usage,
                'ram_percent': ram_percent,
                'ram_available_gb': ram_available,
                'gpu_underused': gpu_percent < self.target_gpu_usage,
                'ram_overused': ram_percent > 90
            }

            return status
        return None

    def print_optimization_report(self):
        """Imprime relatório de otimização"""
        status = self.check_status()
        if not status:
            print("❌ CUDA not available")
            return

        print("=" * 60)
        print("🚀 GPU OPTIMIZATION REPORT")
        print("=" * 60)
        print(
            f"GPU Usage: {status['gpu_allocated_gb']:.2f}GB / {status['gpu_total_gb']:.2f}GB ({status['gpu_percent']:.1f}%)")
        print(f"GPU Target: {status['gpu_target']}%")
        print(f"RAM Usage: {status['ram_percent']:.1f}%")
        print(f"RAM Available: {status['ram_available_gb']:.1f}GB")

        if status['gpu_underused']:
            print("\n⚠️  GPU UNDERUTILIZED!")
            print("💡 Suggestions:")
            print("   - Increase resolution (up to 640x832)")
            print("   - Use higher precision (bfloat16)")
            print("   - Increase batch size")
            print("   - Keep all models on GPU")
        else:
            print("\n✅ GPU well utilized!")

        if status['ram_overused']:
            print("\n⚠️  RAM OVERUSED!")
            print("💡 Suggestions:")
            print("   - Move more processing to GPU")
            print("   - Disable CPU offloading")
            print("   - Use device_map='cuda'")
        else:
            print("\n✅ RAM usage acceptable")

        print("=" * 60)


# Instância global
gpu_monitor = GPUMonitor()
