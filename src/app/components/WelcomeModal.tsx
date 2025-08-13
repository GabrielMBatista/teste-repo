'use client'

import React from 'react'
import { Play, Eye, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface WelcomeModalProps {
  isOpen: boolean
  onClose: () => void
  onStartTour: () => void
  onHighlightMode: () => void
  isDarkMode: boolean
}

export default function WelcomeModal({ 
  isOpen, 
  onClose, 
  onStartTour, 
  onHighlightMode, 
  isDarkMode 
}: WelcomeModalProps) {
  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backgroundColor: 'rgba(0, 0, 0, 0.6)' }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="welcome-title"
      aria-describedby="welcome-description"
    >
      <div
        className={`relative w-full max-w-md rounded-xl shadow-2xl transition-all duration-300 transform ${
          isOpen ? 'scale-100 opacity-100' : 'scale-95 opacity-0'
        } ${
          isDarkMode
            ? 'bg-gray-800 border border-gray-600'
            : 'bg-white border border-gray-200'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className={`absolute top-4 right-4 p-2 rounded-full transition-colors ${
            isDarkMode
              ? 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
              : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
          }`}
          aria-label="Fechar modal"
        >
          <X size={20} />
        </button>

        {/* Content */}
        <div className="p-8">
          {/* Header */}
          <div className="text-center mb-6">
            <div
              className={`w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center ${
                isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
              }`}
            >
              <span className="text-2xl font-bold text-white">G•1</span>
            </div>
            <h2
              id="welcome-title"
              className={`text-2xl font-bold mb-2 ${
                isDarkMode ? 'text-gray-100' : 'text-gray-900'
              }`}
            >
              Bem-vindo ao G•One!
            </h2>
            <p
              id="welcome-description"
              className={`text-base leading-relaxed ${
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}
            >
              Seu assistente virtual inteligente está pronto para ajudar. 
              Descubra como navegar e aproveitar ao máximo todas as funcionalidades disponíveis.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="space-y-3">
            <Button
              onClick={onStartTour}
              className={`w-full flex items-center justify-center gap-3 py-3 text-base font-medium transition-all duration-200 ${
                isDarkMode
                  ? 'bg-blue-600 hover:bg-blue-700 text-white'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              }`}
            >
              <Play size={20} />
              Iniciar Tour Guiado
            </Button>

            <Button
              onClick={onHighlightMode}
              variant="outline"
              className={`w-full flex items-center justify-center gap-3 py-3 text-base font-medium transition-all duration-200 ${
                isDarkMode
                  ? 'border-gray-600 text-gray-200 hover:bg-gray-700 hover:border-gray-500'
                  : 'border-gray-300 text-gray-700 hover:bg-gray-50 hover:border-gray-400'
              }`}
            >
              <Eye size={20} />
              Modo Destacado
            </Button>

            <Button
              onClick={onClose}
              variant="ghost"
              className={`w-full py-3 text-base font-medium transition-all duration-200 ${
                isDarkMode
                  ? 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
              }`}
            >
              Fechar
            </Button>
          </div>

          {/* Footer */}
          <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-600">
            <p
              className={`text-sm text-center ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}
            >
              Você pode acessar essas opções novamente através das configurações
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}