# G•One Chatbot Interface

## 🤖 Sobre o Projeto

Interface completa do chatbot G•One construída com Next.js, TypeScript e TailwindCSS. 

### ✨ Funcionalidades

- **Widget Flutuante Unificado** - Chat draggável que se transforma em interface completa
- **Sistema Avançado de Destaque** - Spotlight com overlay e tooltips informativos
- **Tour Guiado** - Sistema de onboarding multi-etapas
- **Configuração do Bot** - Personalização de nome, personalidade, avatar e modelo
- **Modo Escuro/Claro** - Alternância completa de temas
- **Design Responsivo** - Funciona em todos os dispositivos

### 🚀 Componentes Principais

- `AdvancedHighlightSystem` - Sistema de destaque com spotlight
- `UnifiedChatWidget` - Widget de chat unificado
- `ChatBox` - Interface de chat com histórico
- `FloatingWidget` - Widget flutuante draggável
- `BotConfig` - Tela de configurações
- `HighlightMode` - Modo de destaque de elementos
- `GuidedTour` - Tour guiado interativo
- `WelcomeModal` - Modal de boas-vindas

## 🛠️ Tecnologias

- **Next.js 15** - Framework React com App Router
- **TypeScript** - Tipagem estática
- **TailwindCSS** - Utility-first CSS
- **Lucide React** - Ícones modernos
- **shadcn/ui** - Componentes de UI

## 🚀 Como Executar

Primeiro, instale as dependências:

```bash
npm install
```

Em seguida, execute o servidor de desenvolvimento:

```bash
npm run dev
# ou
yarn dev
# ou
pnpm dev
# ou
bun dev
```

Abra [http://localhost:3000](http://localhost:3000) no seu navegador para ver o resultado.

## 📁 Estrutura do Projeto

```
src/
├── app/
│   ├── components/          # Componentes principais do G•One
│   ├── page.tsx            # Interface principal
│   ├── layout.tsx          # Layout da aplicação
│   └── globals.css         # Estilos globais e temas
├── components/ui/          # Componentes de UI reutilizáveis
└── lib/
    └── utils.ts           # Funções utilitárias
```

## 🎨 Temas

O projeto suporta alternância completa entre modo claro e escuro com:
- 18+ variáveis CSS customizáveis
- Persistência de preferência no localStorage
- Detecção automática da preferência do sistema

## 📱 Responsividade

- **Mobile First** - Design otimizado para dispositivos móveis
- **Posicionamento Dinâmico** - Chat se adapta ao espaço disponível
- **Interações Touch** - Suporte completo a gestos mobile

## 🔧 Builds e Deploy

Para fazer o build de produção:

```bash
npm run build
```

Para fazer deploy, recomendamos usar a [Plataforma Vercel](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme).

## 📚 Saiba Mais

- [Documentação do Next.js](https://nextjs.org/docs)
- [Tutorial Interativo do Next.js](https://nextjs.org/learn)
- [Repositório do Next.js no GitHub](https://github.com/vercel/next.js)

---

Este projeto foi criado com [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app) e otimizado para o G•One chatbot.
