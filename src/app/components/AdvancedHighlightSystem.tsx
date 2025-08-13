'use client'

import React, { useState, useEffect, useRef } from 'react'
import { X, Info, Eye, Target } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface HighlightSystemProps {
  isActive: boolean
  onClose: () => void
  isDarkMode: boolean
}

interface HighlightedElement {
  element: HTMLElement
  originalOutline: string
  originalPosition: string
  originalZIndex: string
  rect: DOMRect
}

export default function AdvancedHighlightSystem({ isActive, onClose, isDarkMode }: HighlightSystemProps) {
  const [highlightedElements, setHighlightedElements] = useState<HighlightedElement[]>([])
  const [selectedElement, setSelectedElement] = useState<HTMLElement | null>(null)
  const [selectedRect, setSelectedRect] = useState<DOMRect | null>(null)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
  const [showTooltip, setShowTooltip] = useState(false)
  const [showSpotlight, setShowSpotlight] = useState(false)
  const overlayRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!isActive) {
      // Limpar highlights quando desativar
      highlightedElements.forEach(({ element, originalOutline, originalPosition, originalZIndex }) => {
        element.style.outline = originalOutline
        element.style.position = originalPosition
        element.style.zIndex = originalZIndex
        element.removeEventListener('click', handleElementClick)
      })
      setHighlightedElements([])
      setShowTooltip(false)
      setShowSpotlight(false)
      setSelectedElement(null)
      return
    }

    // Encontrar e destacar elementos com data-gabs
    const elements = document.querySelectorAll('[data-gabs]') as NodeList
    const highlighted: HighlightedElement[] = []

    elements.forEach((element) => {
      const htmlElement = element as HTMLElement
      const originalOutline = htmlElement.style.outline
      const originalPosition = htmlElement.style.position
      const originalZIndex = htmlElement.style.zIndex
      const rect = htmlElement.getBoundingClientRect()

      // Aplicar highlight inicial
      htmlElement.style.outline = `3px dashed ${isDarkMode ? '#60a5fa' : '#0028af'}`
      htmlElement.style.position = htmlElement.style.position || 'relative'
      htmlElement.style.zIndex = '10'
      
      // Adicionar event listener
      htmlElement.addEventListener('click', handleElementClick)
      htmlElement.style.cursor = 'pointer'

      // Adicionar animação de pulso
      htmlElement.style.animation = 'highlight-pulse 2s infinite'

      highlighted.push({
        element: htmlElement,
        originalOutline,
        originalPosition,
        originalZIndex,
        rect
      })
    })

    setHighlightedElements(highlighted)

    // Adicionar CSS para animações
    const style = document.createElement('style')
    style.textContent = `
      @keyframes highlight-pulse {
        0%, 100% { 
          box-shadow: 0 0 0 0 ${isDarkMode ? 'rgba(96, 165, 250, 0.7)' : 'rgba(0, 40, 175, 0.7)'}; 
        }
        50% { 
          box-shadow: 0 0 0 8px ${isDarkMode ? 'rgba(96, 165, 250, 0.2)' : 'rgba(0, 40, 175, 0.2)'}; 
        }
      }
      
      @keyframes spotlight-glow {
        0% { 
          border-width: 3px;
          box-shadow: 0 0 20px ${isDarkMode ? 'rgba(96, 165, 250, 0.4)' : 'rgba(0, 40, 175, 0.4)'};
        }
        100% { 
          border-width: 4px;
          box-shadow: 0 0 40px ${isDarkMode ? 'rgba(96, 165, 250, 0.8)' : 'rgba(0, 40, 175, 0.8)'};
        }
      }
      
      @keyframes spotlight-indicator {
        0%, 100% { 
          transform: scale(1);
          box-shadow: 0 0 20px ${isDarkMode ? 'rgba(96, 165, 250, 0.5)' : 'rgba(0, 40, 175, 0.5)'};
        }
        50% { 
          transform: scale(1.1);
          box-shadow: 0 0 30px ${isDarkMode ? 'rgba(96, 165, 250, 0.8)' : 'rgba(0, 40, 175, 0.8)'};
        }
      }
    `
    document.head.appendChild(style)

    // Cleanup
    return () => {
      highlighted.forEach(({ element, originalOutline, originalPosition, originalZIndex }) => {
        element.style.outline = originalOutline
        element.style.position = originalPosition
        element.style.zIndex = originalZIndex
        element.style.cursor = ''
        element.style.animation = ''
        element.removeEventListener('click', handleElementClick)
      })
      document.head.removeChild(style)
    }
  }, [isActive, isDarkMode])

  const handleElementClick = (event: Event) => {
    event.preventDefault()
    event.stopPropagation()
    
    const element = event.currentTarget as HTMLElement
    const rect = element.getBoundingClientRect()
    
    // Remover animação de pulso de todos os elementos
    highlightedElements.forEach(({ element: el }) => {
      el.style.animation = ''
      el.style.outline = `2px solid ${isDarkMode ? '#475569' : '#e2e8f0'}`
    })

    // Destacar elemento selecionado
    element.style.outline = `3px solid ${isDarkMode ? '#60a5fa' : '#0028af'}`
    element.style.zIndex = '1000'
    
    setSelectedElement(element)
    setSelectedRect(rect)
    setShowSpotlight(true)
    
    // Calcular posição do tooltip
    const tooltipX = rect.left + rect.width / 2
    const tooltipY = rect.bottom + 16
    
    setTooltipPosition({ x: tooltipX, y: tooltipY })
    setShowTooltip(true)

    // Auto-fechar após 8 segundos
    setTimeout(() => {
      handleCloseSpotlight()
    }, 8000)
  }

  const handleCloseSpotlight = () => {
    setShowSpotlight(false)
    setShowTooltip(false)
    setSelectedElement(null)
    setSelectedRect(null)
    
    // Restaurar animação de pulso
    highlightedElements.forEach(({ element }) => {
      element.style.outline = `3px dashed ${isDarkMode ? '#60a5fa' : '#0028af'}`
      element.style.animation = 'highlight-pulse 2s infinite'
      element.style.zIndex = '10'
    })
  }

  const getElementInfo = (element: HTMLElement) => {
    const gabsValue = element.getAttribute('data-gabs')
    const tagName = element.tagName.toLowerCase()
    const className = element.className
    const textContent = element.textContent?.slice(0, 100) + (element.textContent && element.textContent.length > 100 ? '...' : '')
    const id = element.id

    return {
      gabsValue,
      tagName,
      className,
      textContent,
      id
    }
  }

  if (!isActive) return null

  return (
    <>
      {/* Notification Banner */}
      <div
        className={`fixed top-4 left-1/2 transform -translate-x-1/2 z-50 px-6 py-4 rounded-lg shadow-lg border transition-all duration-300 ${
          isDarkMode
            ? 'bg-slate-800 border-slate-600 text-slate-100'
            : 'bg-white border-gray-200 text-gray-900'
        }`}
      >
        <div className="flex items-center gap-4">
          <div className={`p-2 rounded-full ${isDarkMode ? 'bg-blue-600' : 'bg-blue-500'}`}>
            <Target size={20} className="text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-sm">Modo Destacado Ativo</h3>
            <p className="text-xs opacity-80">
              {highlightedElements.length} elementos encontrados. Clique nos elementos destacados para mais informações.
            </p>
          </div>
          <Button
            onClick={onClose}
            variant="ghost"
            size="sm"
            className={`ml-2 p-2 ${
              isDarkMode
                ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
            }`}
          >
            <X size={16} />
          </Button>
        </div>
      </div>

      {/* Full Page Overlay with Spotlight Effect */}
      {showSpotlight && selectedRect && (
        <div
          ref={overlayRef}
          className="fixed inset-0 z-40 transition-all duration-500"
          onClick={handleCloseSpotlight}
        >
          {/* Dark overlay covering entire page */}
          <div 
            className="absolute inset-0 transition-all duration-500"
            style={{
              backgroundColor: isDarkMode ? 'rgba(0, 0, 0, 0.85)' : 'rgba(0, 0, 0, 0.75)',
              backdropFilter: 'blur(2px)'
            }}
          />
          
          {/* Spotlight cutout - creates the "hole" effect */}
          <div
            className="absolute transition-all duration-500"
            style={{
              left: selectedRect.left - 12,
              top: selectedRect.top - 12,
              width: selectedRect.width + 24,
              height: selectedRect.height + 24,
              background: 'transparent',
              borderRadius: '8px',
              boxShadow: `
                0 0 0 ${selectedRect.left + 12}px ${isDarkMode ? 'rgba(0, 0, 0, 0.85)' : 'rgba(0, 0, 0, 0.75)'}, 
                0 0 0 ${window.innerWidth}px ${isDarkMode ? 'rgba(0, 0, 0, 0.85)' : 'rgba(0, 0, 0, 0.75)'},
                inset 0 0 0 3px ${isDarkMode ? '#60a5fa' : '#0028af'},
                inset 0 0 20px ${isDarkMode ? 'rgba(96, 165, 250, 0.3)' : 'rgba(0, 40, 175, 0.3)'},
                0 0 30px ${isDarkMode ? 'rgba(96, 165, 250, 0.6)' : 'rgba(0, 40, 175, 0.6)'}
              `,
              zIndex: 41
            }}
          />
          
          {/* Animated border around spotlight */}
          <div
            className="absolute border-4 rounded-lg animate-pulse"
            style={{
              left: selectedRect.left - 16,
              top: selectedRect.top - 16,
              width: selectedRect.width + 32,
              height: selectedRect.height + 32,
              borderColor: isDarkMode ? '#60a5fa' : '#0028af',
              borderStyle: 'solid',
              background: 'transparent',
              zIndex: 42,
              animation: 'spotlight-glow 2s ease-in-out infinite alternate'
            }}
          />
          
          {/* Spotlight indicator */}
          <div
            className="absolute flex items-center justify-center z-50"
            style={{
              left: Math.min(selectedRect.right + 20, window.innerWidth - 60),
              top: Math.max(selectedRect.top - 8, 20),
            }}
          >
            <div 
              className={`p-3 rounded-full shadow-2xl border-2 ${
                isDarkMode 
                  ? 'bg-blue-600 border-blue-400 shadow-blue-500/50' 
                  : 'bg-blue-500 border-blue-300 shadow-blue-600/50'
              }`}
              style={{
                animation: 'spotlight-indicator 1.5s ease-in-out infinite'
              }}
            >
              <Eye size={18} className="text-white" />
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Tooltip */}
      {showTooltip && selectedElement && (
        <div
          className="fixed z-[60] transition-all duration-300 transform"
          style={{
            left: Math.min(
              Math.max(16, tooltipPosition.x - 200), 
              window.innerWidth - 416
            ),
            top: Math.min(
              tooltipPosition.y, 
              window.innerHeight - 300
            ),
          }}
        >
          <div
            className={`w-96 max-w-[90vw] rounded-xl shadow-2xl border backdrop-blur-sm ${
              isDarkMode
                ? 'bg-slate-800/95 border-slate-600'
                : 'bg-white/95 border-gray-200'
            }`}
          >
            {/* Header */}
            <div className={`p-4 border-b ${isDarkMode ? 'border-slate-600' : 'border-gray-200'}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${isDarkMode ? 'bg-blue-600' : 'bg-blue-500'}`}>
                    <Info size={16} className="text-white" />
                  </div>
                  <div>
                    <h3 className={`font-semibold text-sm ${isDarkMode ? 'text-slate-100' : 'text-gray-900'}`}>
                      Elemento em Destaque
                    </h3>
                    <p className={`text-xs ${isDarkMode ? 'text-slate-400' : 'text-gray-500'}`}>
                      Informações detalhadas
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleCloseSpotlight}
                  className={`p-2 rounded-full transition-colors ${
                    isDarkMode
                      ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                  }`}
                  aria-label="Fechar spotlight"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="p-4 space-y-4">
              {(() => {
                const info = getElementInfo(selectedElement)
                return (
                  <>
                    {/* Data-gabs value */}
                    <div className="flex items-center gap-3">
                      <span className={`text-xs font-medium ${isDarkMode ? 'text-slate-300' : 'text-gray-700'}`}>
                        Data-gabs:
                      </span>
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-mono ${
                          isDarkMode
                            ? 'bg-blue-600 text-white'
                            : 'bg-blue-100 text-blue-700'
                        }`}
                      >
                        {info.gabsValue || 'N/A'}
                      </span>
                    </div>
                    
                    {/* Element details */}
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div>
                        <span className={`font-medium block ${isDarkMode ? 'text-slate-300' : 'text-gray-700'}`}>
                          Tag:
                        </span>
                        <span className={`${isDarkMode ? 'text-slate-400' : 'text-gray-600'}`}>
                          &lt;{info.tagName}&gt;
                        </span>
                      </div>
                      
                      {info.id && (
                        <div>
                          <span className={`font-medium block ${isDarkMode ? 'text-slate-300' : 'text-gray-700'}`}>
                            ID:
                          </span>
                          <span className={`font-mono ${isDarkMode ? 'text-slate-400' : 'text-gray-600'}`}>
                            #{info.id}
                          </span>
                        </div>
                      )}
                    </div>

                    {info.className && (
                      <div>
                        <span className={`text-xs font-medium block mb-1 ${isDarkMode ? 'text-slate-300' : 'text-gray-700'}`}>
                          Classes:
                        </span>
                        <div className={`text-xs font-mono p-2 rounded ${
                          isDarkMode ? 'bg-slate-700/50 text-slate-300' : 'bg-gray-100/50 text-gray-700'
                        }`}>
                          {info.className}
                        </div>
                      </div>
                    )}

                    {info.textContent && (
                      <div>
                        <span className={`text-xs font-medium block mb-1 ${isDarkMode ? 'text-slate-300' : 'text-gray-700'}`}>
                          Conteúdo:
                        </span>
                        <p className={`text-xs p-2 rounded ${
                          isDarkMode ? 'bg-slate-700/50 text-slate-300' : 'bg-gray-100/50 text-gray-700'
                        }`}>
                          &quot;{info.textContent}&quot;
                        </p>
                      </div>
                    )}
                  </>
                )
              })()}
            </div>

            {/* Actions */}
            <div className={`p-4 border-t ${isDarkMode ? 'border-slate-600' : 'border-gray-200'}`}>
              <div className="flex gap-2">
                <Button
                  onClick={handleCloseSpotlight}
                  className={`flex-1 ${
                    isDarkMode
                      ? 'bg-blue-600 hover:bg-blue-700 text-white'
                      : 'bg-blue-500 hover:bg-blue-600 text-white'
                  }`}
                  size="sm"
                >
                  <Eye size={14} className="mr-2" />
                  Entendi
                </Button>
                <Button
                  onClick={onClose}
                  variant="outline"
                  size="sm"
                  className={`${
                    isDarkMode
                      ? 'border-slate-600 text-slate-300 hover:bg-slate-700'
                      : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  Sair do Modo
                </Button>
              </div>
            </div>
          </div>

          {/* Arrow pointer */}
          <div
            className={`absolute w-4 h-4 transform rotate-45 ${
              isDarkMode ? 'bg-slate-800 border-l border-t border-slate-600' : 'bg-white border-l border-t border-gray-200'
            }`}
            style={{
              left: Math.min(200, Math.max(20, tooltipPosition.x - (tooltipPosition.x - 200))),
              top: '-8px',
            }}
          />
        </div>
      )}
    </>
  )
}