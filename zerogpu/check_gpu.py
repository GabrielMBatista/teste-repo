#!/usr/bin/env python3

import torch
import subprocess
import sys
import os


def check_gpu():
    print("=== Verificação de GPU ===")

    print(f"PyTorch versão: {torch.__version__}")
    print(f"PyTorch compilado com CUDA: {torch.version.cuda}")
    print(f"CUDA disponível no PyTorch: {torch.cuda.is_available()}")

    # Verificar variáveis de ambiente primeiro
    cuda_vars = [k for k in os.environ.keys() if 'CUDA' in k or 'NVIDIA' in k]
    print("Variáveis CUDA/NVIDIA:", cuda_vars)
    for var in cuda_vars:
        print(f"  {var}={os.environ[var]}")

    # Verificar nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("nvidia-smi funcionando:")
            print(result.stdout)
        else:
            print("nvidia-smi falhou:", result.stderr)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("nvidia-smi não encontrado ou timeout")

    # Verificar bibliotecas CUDA
    try:
        print("Verificando bibliotecas CUDA...")
        result = subprocess.run(
            ['ldconfig', '-p'], capture_output=True, text=True)
        cuda_libs = [line for line in result.stdout.split(
            '\n') if 'cuda' in line.lower()]
        if cuda_libs:
            print("Bibliotecas CUDA encontradas:")
            for lib in cuda_libs[:5]:  # Mostrar apenas as primeiras 5
                print(f"  {lib.strip()}")
        else:
            print("Nenhuma biblioteca CUDA encontrada")
    except Exception as e:
        print(f"Erro ao verificar bibliotecas CUDA: {e}")

    # Remover tentativa de reload que causa erro
    if torch.cuda.is_available():
        print(f"Número de GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memória total: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Capability: {props.major}.{props.minor}")

            # Teste básico de alocação
            try:
                test_tensor = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], device=f'cuda:{i}')
                print(f"  Teste de alocação: OK")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Teste de alocação falhou: {e}")

    print("=== Fim da verificação ===")
    return torch.cuda.is_available()


if __name__ == "__main__":
    gpu_available = check_gpu()
    if not gpu_available:
        print("\nAVISO: GPU não detectada, mas continuando...")
        print("O app tentará usar CPU como fallback.")
    else:
        print("\nGPU detectada com sucesso!")
