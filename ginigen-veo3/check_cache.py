#!/usr/bin/env python3
import os
import sys

def check_cache_directory(cache_path):
    """Verifica o conteúdo do diretório de cache"""
    print(f"Checking cache directory: {cache_path}")
    
    if not os.path.exists(cache_path):
        print(f"❌ Cache directory does not exist: {cache_path}")
        return False
    
    print(f"✅ Cache directory exists: {cache_path}")
    
    # Verificar subdiretórios
    subdirs = ['hub', 'transformers', 'models', 'mmaudio', 'torch']
    for subdir in subdirs:
        subdir_path = os.path.join(cache_path, subdir)
        if os.path.exists(subdir_path):
            file_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
            print(f"  ✅ {subdir}/: {file_count} files")
            
            # Mostrar alguns arquivos como exemplo
            files = os.listdir(subdir_path)[:3]
            for file in files:
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024*1024)
                    print(f"    - {file} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {subdir}/: not found")
    
    # Calcular tamanho total
    total_size = 0
    for root, dirs, files in os.walk(cache_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    
    total_size_gb = total_size / (1024**3)
    print(f"📊 Total cache size: {total_size_gb:.2f} GB")
    
    return True

if __name__ == "__main__":
    # Verificar cache tanto no Windows quanto no WSL
    windows_cache = "E:\\Docker\\ginigen\\cache"
    wsl_cache = "/mnt/e/Docker/ginigen/cache"
    
    print("=" * 50)
    print("VEO3-Free Cache Directory Check")
    print("=" * 50)
    
    if os.path.exists(windows_cache):
        check_cache_directory(windows_cache)
    elif os.path.exists(wsl_cache):
        check_cache_directory(wsl_cache)
    else:
        print("❌ No cache directory found in either location")
        print(f"  - Windows: {windows_cache}")
        print(f"  - WSL: {wsl_cache}")
