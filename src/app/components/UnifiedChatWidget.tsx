'use client'

import React, { useState, useRef, useEffect } from 'react'
import { MessageCircle, Minimize2, Send, Bot, User, Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface Message {
  id: string
  text: string
  isUser: boolean
  timestamp: Date
}

interface UnifiedChatWidgetProps {
  isDarkMode: boolean
  onOpenConfig?: () => void
}

export default function UnifiedChatWidget({ isDarkMode, onOpenConfig }: UnifiedChatWidgetProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [position, setPosition] = useState({ x: window.innerWidth - 80, y: window.innerHeight - 80 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const [isHovered, setIsHovered] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Olá! Sou o G•One, seu assistente virtual. Como posso ajudá-lo hoje?',
      isUser: false,
      timestamp: new Date()
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  
  const widgetRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Persistir posição no localStorage
  useEffect(() => {
    const savedPosition = localStorage.getItem('gone-widget-position')
    if (savedPosition) {
      setPosition(JSON.parse(savedPosition))
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('gone-widget-position', JSON.stringify(position))
  }, [position])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (isOpen) return // Não permitir arrastar quando o chat estiver aberto
    
    if (e.detail === 2) {
      // Duplo clique - focar no input se o chat estiver aberto
      if (isOpen && inputRef.current) {
        inputRef.current.focus()
      }
      return
    }
    
    setIsDragging(true)
    const rect = widgetRef.current?.getBoundingClientRect()
    if (rect) {
      setDragOffset({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      })
    }
  }

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging || isOpen) return
    
    const newX = Math.max(0, Math.min(window.innerWidth - 64, e.clientX - dragOffset.x))
    const newY = Math.max(0, Math.min(window.innerHeight - 64, e.clientY - dragOffset.y))
    
    setPosition({ x: newX, y: newY })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      return () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isDragging, dragOffset, isOpen])

  const handleToggle = () => {
    if (!isDragging) {
      setIsOpen(!isOpen)
      if (!isOpen && inputRef.current) {
        setTimeout(() => inputRef.current?.focus(), 100)
      }
    }
  }

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      isUser: true,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsTyping(true)

    // Simular resposta da IA
    setTimeout(() => {
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Obrigado pela sua pergunta! Estou processando sua solicitação e em breve terei uma resposta detalhada para você.',
        isUser: false,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, aiMessage])
      setIsTyping(false)
    }, 1500)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // Calcular posição do chat quando aberto
  const getChatPosition = () => {
    if (!isOpen) {
      return {
        left: position.x,
        top: position.y,
        width: '64px',
        height: '64px'
      }
    }

    const chatWidth = window.innerWidth >= 1024 ? 720 : Math.min(384, window.innerWidth - 32)
    const chatHeight = 500
    const margin = 16

    // Verificar espaço disponível
    const spaceLeft = position.x
    const spaceRight = window.innerWidth - position.x - 64
    const spaceTop = position.y
    const spaceBottom = window.innerHeight - position.y - 64

    let left = position.x
    let top = position.y

    // Ajustar posição para não sair da tela
    if (position.x + chatWidth > window.innerWidth - margin) {
      left = window.innerWidth - chatWidth - margin
    }
    if (position.y + chatHeight > window.innerHeight - margin) {
      top = window.innerHeight - chatHeight - margin
    }
    if (left < margin) left = margin
    if (top < margin) top = margin

    return {
      left: left,
      top: top,
      width: `${chatWidth}px`,
      height: `${chatHeight}px`
    }
  }

  const chatStyle = getChatPosition()

  return (
    <div
      ref={widgetRef}
      className={`fixed transition-all duration-500 ease-in-out ${
        isDragging && !isOpen ? 'cursor-grabbing' : isOpen ? 'cursor-default' : 'cursor-grab'
      } ${isOpen ? 'z-50' : 'z-40'}`}
      style={{
        left: chatStyle.left,
        top: chatStyle.top,
        width: chatStyle.width,
        height: chatStyle.height,
        transform: `scale(${isHovered && !isOpen ? 1.1 : 1})`,
      }}
      onMouseDown={handleMouseDown}
      onMouseEnter={() => !isOpen && setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {!isOpen ? (
        // Widget fechado
        <div
          onClick={handleToggle}
          className={`w-16 h-16 rounded-full shadow-lg flex items-center justify-center transition-all duration-300 ${
            isDarkMode
              ? 'bg-slate-800 border-2 border-slate-600 text-slate-100'
              : 'bg-white border-2 border-gray-200 text-gray-700'
          } hover:shadow-xl`}
          style={{
            background: isDarkMode 
              ? 'linear-gradient(135deg, #1e293b 0%, #334155 100%)'
              : 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)'
          }}
          role="button"
          tabIndex={0}
          aria-label="Abrir G•One Chatbot"
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              handleToggle()
            }
          }}
        >
          <MessageCircle 
            size={24} 
            className={`transition-all duration-300 animate-pulse ${
              isDarkMode ? 'text-blue-400' : 'text-blue-600'
            }`}
          />
          
          {/* Indicador de notificação */}
          <div
            className={`absolute -top-1 -right-1 w-4 h-4 rounded-full transition-all duration-300 ${
              isDarkMode ? 'bg-blue-400' : 'bg-blue-600'
            } scale-100 animate-bounce`}
          />
        </div>
      ) : (
        // Chat aberto
        <div
          className={`w-full h-full rounded-lg shadow-2xl transition-all duration-500 transform scale-100 opacity-100 ${
            isDarkMode
              ? 'bg-slate-800 border border-slate-600'
              : 'bg-white border border-gray-200'
          }`}
        >
          {/* Header com widget integrado */}
          <div
            className={`flex items-center gap-3 p-4 border-b ${
              isDarkMode
                ? 'border-slate-600 bg-slate-700'
                : 'border-gray-200 bg-gray-50'
            } rounded-t-lg`}
          >
            {/* Widget integrado no header */}
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
                isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
              }`}
            >
              <Bot size={20} className="text-white" />
            </div>
            
            <div className="flex-1">
              <h3
                className={`font-semibold text-sm ${
                  isDarkMode ? 'text-slate-100' : 'text-gray-900'
                }`}
              >
                G•One Assistant
              </h3>
              <p
                className={`text-xs ${
                  isDarkMode ? 'text-slate-400' : 'text-gray-500'
                }`}
              >
                Online agora
              </p>
            </div>

            {/* Botões de ação */}
            <div className="flex gap-2">
              {onOpenConfig && (
                <Button
                  onClick={onOpenConfig}
                  variant="ghost"
                  size="sm"
                  className={`p-2 ${
                    isDarkMode
                      ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-600'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                  }`}
                  aria-label="Configurações"
                >
                  <Settings size={16} />
                </Button>
              )}
              
              <Button
                onClick={handleToggle}
                variant="ghost"
                size="sm"
                className={`p-2 ${
                  isDarkMode
                    ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-600'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
                aria-label="Minimizar chat"
              >
                <Minimize2 size={16} />
              </Button>
            </div>
          </div>

          {/* Messages Area */}
          <div
            className={`flex-1 overflow-y-auto p-4 space-y-4 ${
              isDarkMode ? 'bg-slate-800' : 'bg-white'
            }`}
            style={{ height: 'calc(100% - 140px)' }}
            role="log"
            aria-live="polite"
            aria-label="Histórico de mensagens do chat"
          >
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.isUser ? 'justify-end' : 'justify-start'}`}
              >
                {!message.isUser && (
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
                    }`}
                  >
                    <Bot size={16} className="text-white" />
                  </div>
                )}
                
                <div
                  className={`max-w-[80%] p-3 rounded-lg ${
                    message.isUser
                      ? isDarkMode
                        ? 'bg-blue-600 text-white'
                        : 'bg-blue-500 text-white'
                      : isDarkMode
                      ? 'bg-slate-700 text-slate-100'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <p className="text-sm leading-relaxed">{message.text}</p>
                  <span
                    className={`text-xs mt-1 block ${
                      message.isUser
                        ? 'text-blue-100'
                        : isDarkMode
                        ? 'text-slate-400'
                        : 'text-gray-500'
                    }`}
                  >
                    {message.timestamp.toLocaleTimeString('pt-BR', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </span>
                </div>

                {message.isUser && (
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      isDarkMode ? 'bg-slate-600' : 'bg-gray-300'
                    }`}
                  >
                    <User size={16} className={isDarkMode ? 'text-slate-200' : 'text-gray-600'} />
                  </div>
                )}
              </div>
            ))}

            {isTyping && (
              <div className="flex gap-3 justify-start">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
                  }`}
                >
                  <Bot size={16} className="text-white" />
                </div>
                <div
                  className={`p-3 rounded-lg ${
                    isDarkMode ? 'bg-slate-700' : 'bg-gray-100'
                  }`}
                >
                  <div className="flex gap-1">
                    <div
                      className={`w-2 h-2 rounded-full animate-bounce ${
                        isDarkMode ? 'bg-slate-400' : 'bg-gray-400'
                      }`}
                      style={{ animationDelay: '0ms' }}
                    />
                    <div
                      className={`w-2 h-2 rounded-full animate-bounce ${
                        isDarkMode ? 'bg-slate-400' : 'bg-gray-400'
                      }`}
                      style={{ animationDelay: '150ms' }}
                    />
                    <div
                      className={`w-2 h-2 rounded-full animate-bounce ${
                        isDarkMode ? 'bg-slate-400' : 'bg-gray-400'
                      }`}
                      style={{ animationDelay: '300ms' }}
                    />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div
            className={`p-4 border-t ${
              isDarkMode
                ? 'border-slate-600 bg-slate-700'
                : 'border-gray-200 bg-gray-50'
            } rounded-b-lg`}
          >
            <div className="flex gap-2">
              <Input
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Digite sua pergunta..."
                className={`flex-1 ${
                  isDarkMode
                    ? 'bg-slate-800 border-slate-600 text-slate-100 placeholder-slate-400'
                    : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                }`}
                aria-label="Campo de mensagem"
              />
              <Button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isTyping}
                className={`px-4 ${
                  isDarkMode
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                }`}
                aria-label="Enviar mensagem"
              >
                <Send size={16} />
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}