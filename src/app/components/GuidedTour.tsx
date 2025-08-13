'use client'

import React, { useState, useEffect, useRef } from 'react'
import { ChevronLeft, ChevronRight, X, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface TourStep {
  id: string
  target: string
  title: string
  description: string
  position?: 'top' | 'bottom' | 'left' | 'right'
}

interface GuidedTourProps {
  isActive: boolean
  onComplete: () => void
  onClose: () => void
  isDarkMode: boolean
  steps: TourStep[]
}

export default function GuidedTour({ 
  isActive, 
  onComplete, 
  onClose, 
  isDarkMode, 
  steps 
}: GuidedTourProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
  const [targetElement, setTargetElement] = useState<HTMLElement | null>(null)
  const [isVisible, setIsVisible] = useState(false)
  const tooltipRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!isActive) {
      setIsVisible(false)
      return
    }

    const timer = setTimeout(() => setIsVisible(true), 100)
    return () => clearTimeout(timer)
  }, [isActive])

  useEffect(() => {
    if (!isActive || !steps[currentStep]) return

    const targetSelector = steps[currentStep].target
    const element = document.querySelector(targetSelector) as HTMLElement

    if (element) {
      setTargetElement(element)
      element.scrollIntoView({ behavior: 'smooth', block: 'center' })
      
      // Calcular posição do tooltip
      setTimeout(() => {
        const rect = element.getBoundingClientRect()
        const tooltipRect = tooltipRef.current?.getBoundingClientRect()
        
        let x = rect.left + rect.width / 2
        let y = rect.bottom + 16

        // Ajustar posição para não sair da viewport
        if (tooltipRect) {
          if (x + tooltipRect.width / 2 > window.innerWidth - 16) {
            x = window.innerWidth - tooltipRect.width / 2 - 16
          }
          if (x - tooltipRect.width / 2 < 16) {
            x = tooltipRect.width / 2 + 16
          }
          if (y + tooltipRect.height > window.innerHeight - 16) {
            y = rect.top - tooltipRect.height - 16
          }
        }

        setTooltipPosition({ x, y })
      }, 100)
    } else {
      // Fallback se elemento não for encontrado
      setTargetElement(null)
      setTooltipPosition({ 
        x: window.innerWidth / 2, 
        y: window.innerHeight / 2 
      })
    }
  }, [currentStep, isActive, steps])

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      handleComplete()
    }
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleComplete = () => {
    setIsVisible(false)
    setTimeout(() => {
      onComplete()
      setCurrentStep(0)
    }, 180)
  }

  const handleClose = () => {
    setIsVisible(false)
    setTimeout(() => {
      onClose()
      setCurrentStep(0)
    }, 180)
  }

  if (!isActive || !steps[currentStep]) return null

  const currentStepData = steps[currentStep]

  return (
    <>
      {/* Backdrop com blur */}
      <div
        className={`fixed inset-0 z-40 transition-all duration-300 ${
          isVisible ? 'opacity-100' : 'opacity-0'
        }`}
        style={{
          backgroundColor: 'rgba(0, 0, 0, 0.4)',
          backdropFilter: 'blur(2px)'
        }}
      />

      {/* Spotlight */}
      {targetElement && (
        <div
          className="fixed z-50 pointer-events-none transition-all duration-300"
          style={{
            left: targetElement.getBoundingClientRect().left - 8,
            top: targetElement.getBoundingClientRect().top - 8,
            width: targetElement.getBoundingClientRect().width + 16,
            height: targetElement.getBoundingClientRect().height + 16,
            borderRadius: '8px',
            boxShadow: `0 0 0 4px ${isDarkMode ? '#60a5fa' : '#3b82f6'}, 0 0 0 9999px rgba(0, 0, 0, 0.5)`,
            transform: `scale(${isVisible ? 1 : 0.95})`,
            opacity: isVisible ? 1 : 0
          }}
        />
      )}

      {/* Tooltip */}
      <div
        ref={tooltipRef}
        className={`fixed z-50 w-80 max-w-[90vw] transition-all duration-300 transform ${
          isVisible ? 'scale-100 opacity-100' : 'scale-95 opacity-0'
        }`}
        style={{
          left: tooltipPosition.x,
          top: tooltipPosition.y,
          transform: 'translateX(-50%)'
        }}
      >
        <div
          className={`rounded-lg shadow-2xl border ${
            isDarkMode
              ? 'bg-gray-800 border-gray-600'
              : 'bg-white border-gray-200'
          }`}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 pb-2">
            <div className="flex items-center gap-2">
              <span
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  isDarkMode
                    ? 'bg-blue-600 text-white'
                    : 'bg-blue-100 text-blue-700'
                }`}
              >
                Etapa {currentStep + 1} de {steps.length}
              </span>
            </div>
            <button
              onClick={handleClose}
              className={`p-1 rounded-full transition-colors ${
                isDarkMode
                  ? 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
              }`}
              aria-label="Fechar tour"
            >
              <X size={16} />
            </button>
          </div>

          {/* Content */}
          <div className="px-4 pb-4">
            <h3
              className={`font-semibold text-lg mb-2 ${
                isDarkMode ? 'text-gray-100' : 'text-gray-900'
              }`}
            >
              {currentStepData.title}
            </h3>
            <p
              className={`text-sm leading-relaxed mb-4 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}
            >
              {currentStepData.description}
            </p>

            {/* Controls */}
            <div className="flex items-center justify-between">
              <Button
                onClick={handlePrevious}
                disabled={currentStep === 0}
                variant="outline"
                size="sm"
                className={`flex items-center gap-2 ${
                  isDarkMode
                    ? 'border-gray-600 text-gray-300 hover:bg-gray-700'
                    : 'border-gray-300 text-gray-600 hover:bg-gray-50'
                }`}
              >
                <ChevronLeft size={16} />
                Anterior
              </Button>

              <div className="flex gap-1">
                {steps.map((_, index) => (
                  <div
                    key={index}
                    className={`w-2 h-2 rounded-full transition-colors ${
                      index === currentStep
                        ? isDarkMode ? 'bg-blue-400' : 'bg-blue-500'
                        : isDarkMode ? 'bg-gray-600' : 'bg-gray-300'
                    }`}
                  />
                ))}
              </div>

              <Button
                onClick={handleNext}
                size="sm"
                className={`flex items-center gap-2 ${
                  isDarkMode
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                }`}
              >
                {currentStep === steps.length - 1 ? (
                  <>
                    <Check size={16} />
                    Concluir
                  </>
                ) : (
                  <>
                    Próxima
                    <ChevronRight size={16} />
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Arrow pointer */}
        <div
          className={`absolute w-3 h-3 transform rotate-45 -translate-x-1/2 ${
            isDarkMode ? 'bg-gray-800 border-l border-t border-gray-600' : 'bg-white border-l border-t border-gray-200'
          }`}
          style={{
            left: '50%',
            top: '-6px'
          }}
        />
      </div>
    </>
  )
}