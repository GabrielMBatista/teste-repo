import torch
import time
import gc
import psutil
import threading

class MemoryMonitor:
    def __init__(self, limit_gb=8.0):
        self.limit_gb = limit_gb
        self.monitoring = False
        
    def get_memory_info(self):
        """Retorna informações detalhadas de memória"""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            cpu_memory = psutil.virtual_memory()
            cpu_used = cpu_memory.used / 1024**3
            cpu_total = cpu_memory.total / 1024**3
            
            return {
                'gpu_allocated': gpu_allocated,
                'gpu_reserved': gpu_reserved,
                'gpu_total': gpu_total,
                'gpu_free': gpu_total - gpu_allocated,
                'cpu_used': cpu_used,
                'cpu_total': cpu_total,
                'cpu_percent': cpu_memory.percent
            }
        return None
    
    def check_memory_limit(self):
        """Verifica se está próximo do limite de memória"""
        info = self.get_memory_info()
        if info and info['gpu_allocated'] > self.limit_gb * 0.9:
            print(f"⚠️ GPU memory warning: {info['gpu_allocated']:.2f}GB/{self.limit_gb}GB")
            self.cleanup_memory()
            return True
        return False
    
    def cleanup_memory(self):
        """Limpeza emergencial de memória"""
        print("🧹 Emergency memory cleanup...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
    def start_monitoring(self, interval=5):
        """Inicia monitoramento contínuo"""
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                self.check_memory_limit()
                time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Para o monitoramento"""
        self.monitoring = False

# Instância global do monitor
memory_monitor = MemoryMonitor(limit_gb=8.0)
