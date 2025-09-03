@echo off
echo 🚀 Iniciando WAN GGUF via Docker Desktop...

REM Verificar se Docker Desktop está rodando
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Desktop não está rodando
    echo Por favor, inicie o Docker Desktop primeiro
    pause
    exit /b 1
)

REM Criar diretórios se não existirem
if not exist "E:\Docker\wan\wan-gguf" (
    echo 📁 Criando diretórios...
    mkdir "E:\Docker\wan\wan-gguf\models" 2>nul
    mkdir "E:\Docker\wan\wan-gguf\cache" 2>nul
)

echo 🔨 Construindo e executando...
docker-compose up --build -d

echo ✅ WAN GGUF iniciado!
echo 🌐 Acesse: http://localhost:7860
echo.
echo 📊 Para ver logs: docker-compose logs -f wan-gguf
echo 🛑 Para parar: docker-compose down
echo.
pause
