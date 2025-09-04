import os
import subprocess
import sys
from pathlib import Path

def download_models():
    """Download model files to local directory"""
    
    # Diretório de destino - usar o caminho WSL para acessar o disco E: do Windows
    model_dir = Path("/mnt/e/Docker/wan")
    
    # Criar diretório no Windows se não existir
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Diretório criado/verificado: {model_dir}")
    except PermissionError:
        print(f"Erro de permissão. Tentando criar via Windows...")
        # Fallback: criar via comando Windows
        subprocess.run(["cmd.exe", "/c", "mkdir", "E:\\Docker\\wan"], capture_output=True)
        
    print(f"Baixando modelos para: {model_dir}")
    
    # Verificar se estamos no WSL
    if not Path("/mnt/e").exists():
        print("AVISO: Não foi possível acessar /mnt/e. Verifique se o WSL tem acesso aos discos do Windows.")
        print("Execute: sudo mkdir -p /mnt/e && sudo mount -t drvfs E: /mnt/e")
        return
    
    # Usar huggingface-hub para baixar apenas os arquivos necessários
    try:
        # Instalar dependências se necessário
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"], check=True)
        
        from huggingface_hub import snapshot_download
        
        # Baixar todos os arquivos primeiro para entender a estrutura
        print("Baixando todos os arquivos para análise...")
        snapshot_download(
            repo_id="zerogpu-aoti/wan2-2-fp8da-aoti-faster",
            repo_type="space",
            local_dir=str(model_dir),
            resume_download=True
        )
        
        print("Download concluído!")
        print(f"Verifique os arquivos em: {model_dir}")
        
        # Listar estrutura de arquivos baixados
        for root, dirs, files in os.walk(model_dir):
            level = root.replace(str(model_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
    except Exception as e:
        print(f"Erro durante o download: {e}")
        print("Tentando com git-lfs...")
        
        # Fallback usando git - usar diretório WSL correto
        current_dir = os.getcwd()
        os.chdir(str(model_dir.parent))
        subprocess.run([
            "git", "clone", 
            "https://huggingface.co/spaces/zerogpu-aoti/wan2-2-fp8da-aoti-faster",
            "wan"
        ], check=True)
        os.chdir(current_dir)

if __name__ == "__main__":
    download_models()
