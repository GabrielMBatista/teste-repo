'use client'

import React, { useState, useRef, useEffect } from 'react'
import { MessageCircle, Minimize2 } from 'lucide-react'

interface FloatingWidgetProps {
  isOpen: boolean
  onToggle: () => void
  onDoubleClick: () => void
  isDarkMode: boolean
  onPositionChange?: (position: { x: number; y: number }) => void
}

export default function FloatingWidget({ isOpen, onToggle, onDoubleClick, isDarkMode, onPositionChange }: FloatingWidgetProps) {
  const [position, setPosition] = useState({ x: window.innerWidth - 80, y: window.innerHeight - 80 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const [isHovered, setIsHovered] = useState(false)
  const widgetRef = useRef<HTMLDivElement>(null)

  // Persistir posição no localStorage
  useEffect(() => {
    const savedPosition = localStorage.getItem('gone-widget-position')
    if (savedPosition) {
      setPosition(JSON.parse(savedPosition))
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('gone-widget-position', JSON.stringify(position))
    onPositionChange?.(position)
  }, [position, onPositionChange])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.detail === 2) {
      onDoubleClick()
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
    if (!isDragging) return
    
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
  }, [isDragging, dragOffset])

  const handleClick = (e: React.MouseEvent) => {
    if (!isDragging) {
      onToggle()
    }
  }

  return (
    <div
      ref={widgetRef}
      className={`fixed z-50 transition-all duration-300 cursor-pointer select-none ${
        isDragging ? 'cursor-grabbing' : 'cursor-grab'
      }`}
      style={{
        left: position.x,
        top: position.y,
        transform: `scale(${isHovered ? 1.1 : 1})`,
      }}
      onMouseDown={handleMouseDown}
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      role="button"
      tabIndex={0}
      aria-label="Abrir G•One Chatbot"
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          onToggle()
        }
      }}
    >
      <div
        className={`w-16 h-16 rounded-full shadow-lg flex items-center justify-center transition-all duration-300 ${
          isDarkMode
            ? 'bg-slate-800 border-2 border-slate-600 text-slate-100'
            : 'bg-white border-2 border-gray-200 text-gray-700'
        } ${isOpen ? 'rotate-180' : ''} hover:shadow-xl`}
        style={{
          background: isDarkMode 
            ? 'linear-gradient(135deg, #1e293b 0%, #334155 100%)'
            : 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)'
        }}
      >
        {isOpen ? (
          <Minimize2 
            size={24} 
            className={`transition-all duration-300 ${
              isDarkMode ? 'text-blue-400' : 'text-blue-600'
            }`}
          />
        ) : (
          <MessageCircle 
            size={24} 
            className={`transition-all duration-300 animate-pulse ${
              isDarkMode ? 'text-blue-400' : 'text-blue-600'
            }`}
          />
        )}
      </div>
      
      {/* Indicador de notificação */}
      <div
        className={`absolute -top-1 -right-1 w-4 h-4 rounded-full transition-all duration-300 ${
          isDarkMode ? 'bg-blue-400' : 'bg-blue-600'
        } ${isOpen ? 'scale-0' : 'scale-100 animate-bounce'}`}
      />
    </div>
  )
}