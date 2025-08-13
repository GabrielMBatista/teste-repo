'use client'

import React, { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { Moon, Sun, Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'

const UnifiedChatWidget = dynamic(() => import('./components/UnifiedChatWidget'), { ssr: false })
const WelcomeModal = dynamic(() => import('./components/WelcomeModal'), { ssr: false })
const GuidedTour = dynamic(() => import('./components/GuidedTour'), { ssr: false })
const HighlightMode = dynamic(() => import('./components/HighlightMode'), { ssr: false })
const BotConfig = dynamic(() => import('./components/BotConfig'), { ssr: false })

interface TourStep {
  id: string
  target: string
  title: string
  description: string
}

interface BotConfiguration {
  name: string
  personality: string
  avatar: string
  model: string
}

const tourSteps: TourStep[] = [
  {
    id: '1',
    target: '[data-gabs="floating-widget"]',
    title: 'Widget Flutuante',
    description: 'Este é o G•One! Clique para abrir o chat ou arraste para reposicionar. Duplo clique foca no campo de pergunta.'
  },
  {
    id: '2',
    target: '[data-gabs="chat-input"]',
    title: 'Campo de Mensagem',
    description: 'Digite suas perguntas aqui. Pressione Enter para enviar ou use o botão ao lado.'
  },
  {
    id: '3',
    target: '[data-gabs="chat-history"]',
    title: 'Histórico de Conversas',
    description: 'Aqui você verá todo o histórico da sua conversa com o G•One, organizado de forma clara.'
  },
  {
    id: '4',
    target: '[data-gabs="theme-toggle"]',
    title: 'Alternar Tema',
    description: 'Alterne entre modo claro e escuro para uma experiência visual personalizada.'
  },
  {
    id: '5',
    target: '[data-gabs="config-button"]',
    title: 'Configurações',
    description: 'Acesse as configurações para personalizar seu bot: nome, personalidade, avatar e modelo de IA.'
  }
]

export default function GOneInterface() {
  const [isDarkMode, setIsDarkMode] = useState(false)
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [showWelcome, setShowWelcome] = useState(true)
  const [showTour, setShowTour] = useState(false)
  const [showHighlight, setShowHighlight] = useState(false)
  const [showConfig, setShowConfig] = useState(false)
  const [focusInput, setFocusInput] = useState(false)
  const [widgetPosition, setWidgetPosition] = useState({ x: 0, y: 0 })

  // Detectar preferência do sistema para dark mode
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const savedTheme = localStorage.getItem('gone-theme')
      if (savedTheme) {
        setIsDarkMode(savedTheme === 'dark')
      } else {
        setIsDarkMode(window.matchMedia('(prefers-color-scheme: dark)').matches)
      }

      // Verificar se é primeira visita
      const hasVisited = localStorage.getItem('gone-visited')
      if (hasVisited) {
        setShowWelcome(false)
      }
    }
  }, [])

  // Aplicar tema ao documento
  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode)
    localStorage.setItem('gone-theme', isDarkMode ? 'dark' : 'light')
  }, [isDarkMode])

  const handleToggleTheme = () => {
    setIsDarkMode(!isDarkMode)
  }

  const handleToggleChat = () => {
    setIsChatOpen(!isChatOpen)
  }

  const handleDoubleClickWidget = () => {
    if (!isChatOpen) {
      setIsChatOpen(true)
    }
    setFocusInput(true)
    setTimeout(() => setFocusInput(false), 100)
  }

  const handleCloseWelcome = () => {
    setShowWelcome(false)
    localStorage.setItem('gone-visited', 'true')
  }

  const handleStartTour = () => {
    setShowWelcome(false)
    setShowTour(true)
    localStorage.setItem('gone-visited', 'true')
  }

  const handleStartHighlight = () => {
    setShowWelcome(false)
    setShowHighlight(true)
    localStorage.setItem('gone-visited', 'true')
  }

  const handleCompleteTour = () => {
    setShowTour(false)
  }

  const handleCloseTour = () => {
    setShowTour(false)
  }

  const handleCloseHighlight = () => {
    setShowHighlight(false)
  }

  const handleOpenConfig = () => {
    setShowConfig(true)
  }

  const handleCloseConfig = () => {
    setShowConfig(false)
  }

  const handleSaveConfig = (config: BotConfiguration) => {
    console.log('Configuração salva:', config)
    localStorage.setItem('gone-config', JSON.stringify(config))
  }

  const handleWidgetPositionChange = (position: { x: number; y: number }) => {
    setWidgetPosition(position)
  }

  return (
    <div
      className={`min-h-screen transition-colors duration-300 ${
        isDarkMode
          ? 'bg-slate-900 text-slate-100'
          : 'bg-gray-50 text-gray-900'
      }`}
    >
      {/* Header com controles */}
      <header
        className={`fixed top-0 left-0 right-0 z-30 p-4 border-b backdrop-blur-sm ${
          isDarkMode
            ? 'bg-slate-900/80 border-slate-700'
            : 'bg-white/80 border-gray-200'
        }`}
      >
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-3">
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center ${
                isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
              }`}
            >
              <span className="text-sm font-bold text-white">G•1</span>
            </div>
            <h1 className="text-lg font-semibold">G•One Assistant</h1>
          </div>

          <div className="flex items-center gap-2">
            <Button
              onClick={handleOpenConfig}
              variant="ghost"
              size="sm"
              className={`${
                isDarkMode
                  ? 'text-slate-300 hover:text-slate-100 hover:bg-slate-800'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
              data-gabs="config-button"
              aria-label="Abrir configurações"
            >
              <Settings size={18} />
            </Button>

            <Button
              onClick={handleToggleTheme}
              variant="ghost"
              size="sm"
              className={`${
                isDarkMode
                  ? 'text-slate-300 hover:text-slate-100 hover:bg-slate-800'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
              data-gabs="theme-toggle"
              aria-label={isDarkMode ? 'Ativar modo claro' : 'Ativar modo escuro'}
            >
              {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
            </Button>
          </div>
        </div>
      </header>

      {/* Conteúdo principal */}
      <main className="pt-20 pb-8">
        <div className="max-w-7xl mx-auto px-4">
          {/* Área de demonstração com elementos data-gabs */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            <div
              className={`p-6 rounded-lg border ${
                isDarkMode
                  ? 'bg-slate-800 border-slate-700'
                  : 'bg-white border-gray-200'
              }`}
              data-gabs="demo-card-1"
            >
              <h3 className="text-lg font-semibold mb-2">Funcionalidade 1</h3>
              <p
                className={`text-sm ${
                  isDarkMode ? 'text-slate-400' : 'text-gray-600'
                }`}
              >
                Esta é uma área de demonstração que pode ser destacada durante o tour ou modo highlight.
              </p>
            </div>

            <div
              className={`p-6 rounded-lg border ${
                isDarkMode
                  ? 'bg-slate-800 border-slate-700'
                  : 'bg-white border-gray-200'
              }`}
              data-gabs="demo-card-2"
            >
              <h3 className="text-lg font-semibold mb-2">Funcionalidade 2</h3>
              <p
                className={`text-sm ${
                  isDarkMode ? 'text-slate-400' : 'text-gray-600'
                }`}
              >
                Outro elemento que pode ser destacado para mostrar recursos específicos da aplicação.
              </p>
            </div>

            <div
              className={`p-6 rounded-lg border ${
                isDarkMode
                  ? 'bg-slate-800 border-slate-700'
                  : 'bg-white border-gray-200'
              }`}
              data-gabs="demo-card-3"
            >
              <h3 className="text-lg font-semibold mb-2">Funcionalidade 3</h3>
              <p
                className={`text-sm ${
                  isDarkMode ? 'text-slate-400' : 'text-gray-600'
                }`}
              >
                Terceiro elemento de demonstração para o sistema de tour e highlight.
              </p>
            </div>
          </div>

          {/* Seção de informações */}
          <div
            className={`p-8 rounded-lg border text-center ${
              isDarkMode
                ? 'bg-slate-800 border-slate-700'
                : 'bg-white border-gray-200'
            }`}
            data-gabs="info-section"
          >
            <h2 className="text-2xl font-bold mb-4">Bem-vindo ao G•One</h2>
            <p
              className={`text-lg mb-6 ${
                isDarkMode ? 'text-slate-300' : 'text-gray-600'
              }`}
            >
              Seu assistente virtual inteligente está pronto para ajudar. 
              Use o widget flutuante para iniciar uma conversa ou explore as funcionalidades através do tour guiado.
            </p>
            
            <div className="flex flex-wrap justify-center gap-4">
              <Button
                onClick={handleStartTour}
                className={`${
                  isDarkMode
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                }`}
              >
                Iniciar Tour
              </Button>
              
              <Button
                onClick={handleStartHighlight}
                variant="outline"
                className={`${
                  isDarkMode
                    ? 'border-slate-600 text-slate-300 hover:bg-slate-700'
                    : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
              >
                Modo Destacado
              </Button>
            </div>
          </div>
        </div>
      </main>

      {/* Unified Chat Widget */}
      <UnifiedChatWidget
        isDarkMode={isDarkMode}
        onOpenConfig={handleOpenConfig}
      />

      {/* Modal de Boas-vindas */}
      <WelcomeModal
        isOpen={showWelcome}
        onClose={handleCloseWelcome}
        onStartTour={handleStartTour}
        onHighlightMode={handleStartHighlight}
        isDarkMode={isDarkMode}
      />

      {/* Tour Guiado */}
      <GuidedTour
        isActive={showTour}
        onComplete={handleCompleteTour}
        onClose={handleCloseTour}
        isDarkMode={isDarkMode}
        steps={tourSteps}
      />

      {/* Modo Destacado */}
      <HighlightMode
        isActive={showHighlight}
        onClose={handleCloseHighlight}
        isDarkMode={isDarkMode}
      />

      {/* Configurações */}
      {showConfig && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ backgroundColor: 'rgba(0, 0, 0, 0.6)' }}
          onClick={handleCloseConfig}
        >
          <div onClick={(e) => e.stopPropagation()}>
            <BotConfig
              isDarkMode={isDarkMode}
              onSave={handleSaveConfig}
              initialConfig={JSON.parse(localStorage.getItem('gone-config') || 'null')}
            />
          </div>
        </div>
      )}
    </div>
  )
}