import time
import torch


class LoadingStatusDisplay:
    def __init__(self):
        self.current_step = 0
        self.total_steps = 0
        self.start_time = None

    def start_loading(self, total_steps, title="Loading"):
        """Inicia o display de carregamento"""
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()

        print("\n" + "=" * 60)
        print(f"🚀 {title.upper()}")
        print("=" * 60)

    def update_step(self, step_name, details=""):
        """Atualiza o step atual"""
        self.current_step += 1

        # Barra de progresso
        progress = self.current_step / self.total_steps
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        percentage = progress * 100

        # Tempo estimado
        elapsed = time.time() - self.start_time
        if progress > 0:
            eta = (elapsed / progress) - elapsed
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --s"

        print(f"\n🔄 STEP {self.current_step}/{self.total_steps}: {step_name}")
        print(f"[{bar}] {percentage:.1f}% | {eta_str}")

        if details:
            print(f"📍 {details}")

        # Mostrar uso de GPU
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_total = torch.cuda.get_device_properties(
                0).total_memory / 1024**3
            print(f"🖥️ GPU: {gpu_allocated:.2f}GB / {gpu_total:.2f}GB")

    def complete(self, final_message="Loading Complete"):
        """Finaliza o display de carregamento"""
        elapsed = time.time() - self.start_time
        print(f"\n✅ {final_message}")
        print(f"⏱️ Total time: {elapsed:.1f}s")
        print("=" * 60)


# Instância global
loading_display = LoadingStatusDisplay()
