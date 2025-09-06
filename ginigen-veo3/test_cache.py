#!/usr/bin/env python3
import os
import tempfile

def test_cache_write():
    """Testa se conseguimos escrever no diretório de cache"""
    cache_dir = "/app/cache"
    
    print(f"Testing cache directory: {cache_dir}")
    
    # Verificar se o diretório existe
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory does not exist: {cache_dir}")
        return False
    
    print(f"✅ Cache directory exists: {cache_dir}")
    
    # Testar escrita
    test_file = os.path.join(cache_dir, "test_write.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("Test cache write")
        print(f"✅ Can write to cache directory")
        
        # Verificar se arquivo foi criado
        if os.path.exists(test_file):
            print(f"✅ File created successfully: {test_file}")
            os.remove(test_file)
            return True
        else:
            print(f"❌ File was not created: {test_file}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot write to cache directory: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Cache Write Test")
    print("=" * 50)
    
    success = test_cache_write()
    
    if success:
        print("✅ Cache is working correctly!")
    else:
        print("❌ Cache has issues!")
        
    # Mostrar variáveis de ambiente relevantes
    print("\nEnvironment variables:")
    env_vars = ['CACHE_DIR', 'HF_HOME', 'TRANSFORMERS_CACHE', 'HF_HUB_CACHE']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"  {var}: {value}")
