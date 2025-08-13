'use client'

import React, { useState, useEffect } from 'react'
import { X, Info } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface HighlightModeProps {
  isActive: boolean
  onClose: () => void
  isDarkMode: boolean
}

interface HighlightedElement {
  element: HTMLElement
  originalOutline: string
  originalPosition: string
}

export default function HighlightMode({ isActive, onClose, isDarkMode }: HighlightModeProps) {
  const [highlightedElements, setHighlightedElements] = useState<HighlightedElement[]>([])
  const [selectedElement, setSelectedElement] = useState<HTMLElement | null>(null)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
  const [showTooltip, setShowTooltip] = useState(false)

  useEffect(() => {
    if (!isActive) {
      // Remover highlights quando desativar
      highlightedElements.forEach(({ element, originalOutline, originalPosition }) => {
        element.style.outline = originalOutline
        element.style.position = originalPosition
        element.removeEventListener('click', handleElementClick)
      })
      setHighlightedElements([])
      setShowTooltip(false)
      return
    }

    // Encontrar e destacar elementos com data-gabs
    const elements = document.querySelectorAll('[data-gabs]') as NodeList
    const highlighted: HighlightedElement[] = []

    elements.forEach((element) => {
      const htmlElement = element as HTMLElement
      const originalOutline = htmlElement.style.outline
      const originalPosition = htmlElement.style.position

      // Aplicar highlight
      htmlElement.style.outline = `2px dashed ${isDarkMode ? '#60a5fa' : '#0028af'}`
      htmlElement.style.position = htmlElement.style.position || 'relative'
      
      // Adicionar event listener
      htmlElement.addEventListener('click', handleElementClick)
      htmlElement.style.cursor = 'pointer'

      highlighted.push({
        element: htmlElement,
        originalOutline,
        originalPosition
      })
    })

    setHighlightedElements(highlighted)

    // Cleanup
    return () => {
      highlighted.forEach(({ element, originalOutline, originalPosition }) => {
        element.style.outline = originalOutline
        element.style.position = originalPosition
        element.style.cursor = ''
        element.removeEventListener('click', handleElementClick)
      })
    }
  }, [isActive, isDarkMode, highlightedElements])

  const handleElementClick = (event: Event) => {
    event.preventDefault()
    event.stopPropagation()
    
    const element = event.currentTarget as HTMLElement
    const rect = element.getBoundingClientRect()
    
    setSelectedElement(element)
    setTooltipPosition({
      x: rect.left + rect.width / 2,
      y: rect.bottom + 8
    })
    setShowTooltip(true)

    // Auto-fechar tooltip após 5 segundos
    setTimeout(() => {
      setShowTooltip(false)
    }, 5000)
  }

  const handleCloseTooltip = () => {
    setShowTooltip(false)
    setSelectedElement(null)
  }

  const getElementInfo = (element: HTMLElement) => {
    const gabsValue = element.getAttribute('data-gabs')
    const tagName = element.tagName.toLowerCase()
    const className = element.className
    const textContent = element.textContent?.slice(0, 50) + (element.textContent && element.textContent.length > 50 ? '...' : '')

    return {
      gabsValue,
      tagName,
      className,
      textContent
    }
  }

  if (!isActive) return null

  return (
    <>
      {/* Notification Banner */}
      <div
        className={`fixed top-4 left-1/2 transform -translate-x-1/2 z-50 px-4 py-3 rounded-lg shadow-lg border transition-all duration-300 ${
          isDarkMode
            ? 'bg-slate-800 border-slate-600 text-slate-100'
            : 'bg-white border-gray-200 text-gray-900'
        }`}
      >
        <div className="flex items-center gap-3">
          <Info size={20} className={isDarkMode ? 'text-blue-400' : 'text-blue-500'} />
          <span className="text-sm font-medium">
            Modo Destacado ativo - Clique nos elementos destacados para mais informações
          </span>
          <Button
            onClick={onClose}
            variant="ghost"
            size="sm"
            className={`ml-2 p-1 ${
              isDarkMode
                ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
            }`}
          >
            <X size={16} />
          </Button>
        </div>
      </div>

      {/* Element Tooltip */}
      {showTooltip && selectedElement && (
        <div
          className={`fixed z-50 w-80 max-w-[90vw] transition-all duration-200 transform ${
            showTooltip ? 'scale-100 opacity-100' : 'scale-95 opacity-0'
          }`}
          style={{
            left: Math.min(tooltipPosition.x, window.innerWidth - 320 - 16),
            top: Math.min(tooltipPosition.y, window.innerHeight - 200),
            transform: tooltipPosition.x > window.innerWidth - 320 - 16 ? 'translateX(-100%)' : 'translateX(-50%)'
          }}
        >
          <div
            className={`rounded-lg shadow-2xl border p-4 ${
              isDarkMode
                ? 'bg-slate-800 border-slate-600'
                : 'bg-white border-gray-200'
            }`}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
              <h3
                className={`font-semibold text-sm ${
                  isDarkMode ? 'text-slate-100' : 'text-gray-900'
                }`}
              >
                Elemento Destacado
              </h3>
              <button
                onClick={handleCloseTooltip}
                className={`p-1 rounded-full transition-colors ${
                  isDarkMode
                    ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
                aria-label="Fechar tooltip"
              >
                <X size={16} />
              </button>
            </div>

            {/* Element Info */}
            <div className="space-y-2 text-xs">
              {(() => {
                const info = getElementInfo(selectedElement)
                return (
                  <>
                    <div>
                      <span
                        className={`font-medium ${
                          isDarkMode ? 'text-slate-300' : 'text-gray-700'
                        }`}
                      >
                        Data-gabs:
                      </span>
                      <span
                        className={`ml-2 px-2 py-1 rounded text-xs font-mono ${
                          isDarkMode
                            ? 'bg-blue-600 text-white'
                            : 'bg-blue-100 text-blue-700'
                        }`}
                      >
                        {info.gabsValue || 'N/A'}
                      </span>
                    </div>
                    
                    <div>
                      <span
                        className={`font-medium ${
                          isDarkMode ? 'text-slate-300' : 'text-gray-700'
                        }`}
                      >
                        Tag:
                      </span>
                      <span
                        className={`ml-2 ${
                          isDarkMode ? 'text-slate-400' : 'text-gray-600'
                        }`}
                      >
                        {info.tagName}
                      </span>
                    </div>

                    {info.className && (
                      <div>
                        <span
                          className={`font-medium ${
                            isDarkMode ? 'text-slate-300' : 'text-gray-700'
                          }`}
                        >
                          Classes:
                        </span>
                        <span
                          className={`ml-2 text-xs font-mono ${
                            isDarkMode ? 'text-slate-400' : 'text-gray-600'
                          }`}
                        >
                          {info.className}
                        </span>
                      </div>
                    )}

                    {info.textContent && (
                      <div>
                        <span
                          className={`font-medium ${
                            isDarkMode ? 'text-slate-300' : 'text-gray-700'
                          }`}
                        >
                          Conteúdo:
                        </span>
                        <p
                          className={`ml-2 mt-1 ${
                            isDarkMode ? 'text-slate-400' : 'text-gray-600'
                          }`}
                        >
                          {info.textContent}
                        </p>
                      </div>
                    )}
                  </>
                )
              })()}
            </div>

            {/* Action Button */}
            <div className="mt-4 pt-3 border-t border-gray-200 dark:border-slate-600">
              <Button
                onClick={handleCloseTooltip}
                size="sm"
                className={`w-full ${
                  isDarkMode
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                }`}
              >
                Ok, entendi
              </Button>
            </div>
          </div>

          {/* Arrow pointer */}
          <div
            className={`absolute w-3 h-3 transform rotate-45 ${
              isDarkMode ? 'bg-slate-800 border-l border-t border-slate-600' : 'bg-white border-l border-t border-gray-200'
            }`}
            style={{
              left: tooltipPosition.x > window.innerWidth - 320 - 16 ? 'calc(100% - 24px)' : '50%',
              top: '-6px',
              transform: tooltipPosition.x > window.innerWidth - 320 - 16 ? 'translateX(-50%) rotate(45deg)' : 'translateX(-50%) rotate(45deg)'
            }}
          />
        </div>
      )}
    </>
  )
}