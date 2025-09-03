# Configuração do Projeto - Postagem Automática Social Media

## 1. Estrutura do Banco de Dados (Supabase)

### Tabela: social_accounts
```sql
CREATE TABLE social_accounts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    platform VARCHAR(20) NOT NULL CHECK (platform IN ('youtube', 'instagram', 'tiktok')),
    account_name VARCHAR(100) NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    token_expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    last_post_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices
CREATE INDEX idx_social_accounts_platform ON social_accounts(platform);
CREATE INDEX idx_social_accounts_active ON social_accounts(is_active);
```

### Tabela: content
```sql
CREATE TABLE content (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES social_accounts(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    media_url TEXT,
    media_type VARCHAR(20) CHECK (media_type IN ('image', 'video', 'carousel')),
    tags TEXT, -- Tags separadas por vírgula
    scheduled_for TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'ready', 'posted', 'error')),
    posted_at TIMESTAMP WITH TIME ZONE,
    post_response JSONB, -- Resposta da API da plataforma
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices
CREATE INDEX idx_content_account_id ON content(account_id);
CREATE INDEX idx_content_status ON content(status);
CREATE INDEX idx_content_scheduled_for ON content(scheduled_for);
CREATE INDEX idx_content_account_status_scheduled ON content(account_id, status, scheduled_for);
```

### Triggers para updated_at
```sql
-- Função para atualizar updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers
CREATE TRIGGER update_social_accounts_updated_at 
    BEFORE UPDATE ON social_accounts 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_updated_at 
    BEFORE UPDATE ON content 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();
```

## 2. Configuração do Arquivo .env

Crie um arquivo `.env` na raiz do seu projeto n8n com as seguintes variáveis:

```env
# Configurações do Supabase
SUPABASE_URL=https://seu-projeto.supabase.co
SUPABASE_ANON_KEY=sua_chave_anonima
SUPABASE_SERVICE_ROLE_KEY=sua_chave_service_role

# Configurações de Conexão PostgreSQL (Supabase)
SUPABASE_DB_HOST=db.seu-projeto.supabase.co
SUPABASE_DB_PORT=5432
SUPABASE_DB_NAME=postgres
SUPABASE_DB_USER=postgres
SUPABASE_DB_PASSWORD=sua_senha_do_banco
SUPABASE_POSTGRES_CREDENTIAL_ID=sua-credential-id-n8n

# YouTube API
YOUTUBE_CLIENT_ID=seu_youtube_client_id
YOUTUBE_CLIENT_SECRET=seu_youtube_client_secret
YOUTUBE_REDIRECT_URI=http://localhost:5678/rest/oauth2-credential/callback

# Instagram/Facebook API
INSTAGRAM_APP_ID=seu_instagram_app_id
INSTAGRAM_APP_SECRET=seu_instagram_app_secret
INSTAGRAM_BUSINESS_ACCOUNT_ID=seu_business_account_id
FACEBOOK_ACCESS_TOKEN=seu_facebook_access_token

# TikTok API
TIKTOK_CLIENT_KEY=sua_tiktok_client_key
TIKTOK_CLIENT_SECRET=sua_tiktok_client_secret

# Configurações do n8n
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=suasenha123
N8N_HOST=localhost
N8N_PORT=5678
N8N_PROTOCOL=http

# Configurações de Log
N8N_LOG_LEVEL=info
N8N_LOG_OUTPUT=console

# Configurações de Execução
N8N_EXECUTIONS_PROCESS=own
N8N_EXECUTIONS_DATA_SAVE_ON_ERROR=all
N8N_EXECUTIONS_DATA_SAVE_ON_SUCCESS=none
N8N_EXECUTIONS_DATA_SAVE_MANUAL_EXECUTIONS=false
```