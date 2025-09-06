import torch
import gc


class MemorySaver:
    def __init__(self):
        self.saved_models = {}

    def aggressive_cleanup(self):
        """Limpeza agressiva de memória"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    def get_memory_mb(self):
        """Retorna memória alocada em MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0

    def check_memory_available(self, required_mb=1000):
        """Verifica se há memória suficiente"""
        if torch.cuda.is_available():
            total_mb = torch.cuda.get_device_properties(
                0).total_memory / 1024 / 1024
            allocated_mb = self.get_memory_mb()
            available_mb = total_mb - allocated_mb

            print(
                f"GPU Memory: {allocated_mb:.0f}MB used, {available_mb:.0f}MB available")
            return available_mb >= required_mb
        return False

    def emergency_cleanup(self):
        """Limpeza emergencial se ficar sem memória"""
        print("🚨 Emergency memory cleanup...")

        # Forçar garbage collection múltiplas vezes
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()

        print(f"After emergency cleanup: {self.get_memory_mb():.0f}MB used")


# Instância global
memory_saver = MemorySaver()
