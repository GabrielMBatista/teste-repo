'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Send, Bot, User } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface Message {
  id: string
  text: string
  isUser: boolean
  timestamp: Date
}

interface ChatBoxProps {
  isOpen: boolean
  isDarkMode: boolean
  onFocusInput?: () => void
  widgetPosition?: { x: number; y: number }
}

export default function ChatBox({ isOpen, isDarkMode, onFocusInput, widgetPosition }: ChatBoxProps) {
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
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    if (onFocusInput && inputRef.current) {
      inputRef.current.focus()
    }
  }, [onFocusInput])

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

  if (!isOpen) return null

  // Calcular posição do chat baseada na posição do widget
  const getChatPosition = () => {
    if (!widgetPosition) {
      return {
        bottom: '20px',
        right: '16px',
        left: 'auto',
        top: 'auto'
      }
    }

    const chatWidth = window.innerWidth >= 1024 ? 720 : 384 // lg:w-[720px] : w-96
    const chatHeight = 500
    const margin = 16

    // Verificar se há espaço à esquerda do widget
    const spaceLeft = widgetPosition.x
    const spaceRight = window.innerWidth - widgetPosition.x - 64 // 64px é a largura do widget
    const spaceTop = widgetPosition.y
    const spaceBottom = window.innerHeight - widgetPosition.y - 64

    let position = {
      bottom: 'auto',
      right: 'auto',
      left: 'auto',
      top: 'auto'
    }

    // Priorizar posicionamento à esquerda do widget se houver espaço
    if (spaceLeft >= chatWidth + margin) {
      position.left = `${Math.max(margin, widgetPosition.x - chatWidth - margin)}px`
      position.top = `${Math.max(margin, Math.min(widgetPosition.y, window.innerHeight - chatHeight - margin))}px`
    }
    // Se não houver espaço à esquerda, tentar à direita
    else if (spaceRight >= chatWidth + margin) {
      position.left = `${widgetPosition.x + 64 + margin}px`
      position.top = `${Math.max(margin, Math.min(widgetPosition.y, window.innerHeight - chatHeight - margin))}px`
    }
    // Se não houver espaço horizontal, posicionar acima ou abaixo
    else if (spaceTop >= chatHeight + margin) {
      position.bottom = `${window.innerHeight - widgetPosition.y + margin}px`
      position.left = `${Math.max(margin, Math.min(widgetPosition.x - chatWidth/2, window.innerWidth - chatWidth - margin))}px`
    }
    else if (spaceBottom >= chatHeight + margin) {
      position.top = `${widgetPosition.y + 64 + margin}px`
      position.left = `${Math.max(margin, Math.min(widgetPosition.x - chatWidth/2, window.innerWidth - chatWidth - margin))}px`
    }
    // Fallback: posição padrão
    else {
      position.bottom = '20px'
      position.right = '16px'
    }

    return position
  }

  const chatPosition = getChatPosition()

  return (
    <div
      className={`fixed w-96 max-w-[90vw] h-[500px] max-h-[70vh] rounded-lg shadow-2xl transition-all duration-300 transform ${
        isOpen ? 'scale-100 opacity-100' : 'scale-95 opacity-0'
      } ${
        isDarkMode
          ? 'bg-slate-800 border border-slate-600'
          : 'bg-white border border-gray-200'
      } lg:w-[720px] lg:max-w-[720px]`}
      style={{
        zIndex: 40,
        ...chatPosition
      }}
    >
      {/* Header */}
      <div
        className={`flex items-center gap-3 p-4 border-b ${
          isDarkMode
            ? 'border-slate-600 bg-slate-700'
            : 'border-gray-200 bg-gray-50'
        } rounded-t-lg`}
      >
        <div
          className={`w-8 h-8 rounded-full flex items-center justify-center ${
            isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
          }`}
        >
          <Bot size={16} className="text-white" />
        </div>
        <div>
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
      </div>

      {/* Messages Area */}
      <div
        className={`flex-1 overflow-y-auto p-4 space-y-4 h-[340px] ${
          isDarkMode ? 'bg-slate-800' : 'bg-white'
        }`}
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
  )
}