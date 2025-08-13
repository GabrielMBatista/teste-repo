'use client'

import React, { useState } from 'react'
import { Save, Upload, User, Brain, Image, Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

interface BotConfigProps {
  isDarkMode: boolean
  onSave: (config: BotConfiguration) => void
  initialConfig?: BotConfiguration
}

interface BotConfiguration {
  name: string
  personality: string
  avatar: string
  model: string
}

export default function BotConfig({ isDarkMode, onSave, initialConfig }: BotConfigProps) {
  const [config, setConfig] = useState<BotConfiguration>(
    initialConfig || {
      name: 'G•One',
      personality: 'Sou um assistente virtual amigável e prestativo, sempre pronto para ajudar com suas dúvidas e tarefas.',
      avatar: '',
      model: 'gpt-4'
    }
  )
  const [isSaving, setIsSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [avatarPreview, setAvatarPreview] = useState<string>(initialConfig?.avatar || '')

  const handleInputChange = (field: keyof BotConfiguration, value: string) => {
    setConfig(prev => ({ ...prev, [field]: value }))
    
    if (field === 'avatar') {
      setAvatarPreview(value)
    }
  }

  const handleAvatarUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result as string
        setAvatarPreview(result)
        handleInputChange('avatar', result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleSave = async () => {
    setIsSaving(true)
    setSaveStatus('idle')

    try {
      // Simular salvamento
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      onSave(config)
      setSaveStatus('success')
      
      setTimeout(() => setSaveStatus('idle'), 3000)
    } catch (error) {
      setSaveStatus('error')
      setTimeout(() => setSaveStatus('idle'), 3000)
    } finally {
      setIsSaving(false)
    }
  }

  const isFormValid = config.name.trim() && config.personality.trim() && config.model

  return (
    <div
      className={`w-full max-w-2xl mx-auto p-6 rounded-lg shadow-lg ${
        isDarkMode
          ? 'bg-slate-800 border border-slate-600'
          : 'bg-white border border-gray-200'
      }`}
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center ${
            isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
          }`}
        >
          <Settings size={20} className="text-white" />
        </div>
        <div>
          <h2
            className={`text-xl font-bold ${
              isDarkMode ? 'text-slate-100' : 'text-gray-900'
            }`}
          >
            Configurações do Bot
          </h2>
          <p
            className={`text-sm ${
              isDarkMode ? 'text-slate-400' : 'text-gray-600'
            }`}
          >
            Personalize seu assistente virtual
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Nome do Bot */}
        <div className="space-y-2">
          <Label
            htmlFor="bot-name"
            className={`flex items-center gap-2 text-sm font-medium ${
              isDarkMode ? 'text-slate-200' : 'text-gray-700'
            }`}
          >
            <User size={16} />
            Nome do Bot
          </Label>
          <Input
            id="bot-name"
            value={config.name}
            onChange={(e) => handleInputChange('name', e.target.value)}
            placeholder="Digite o nome do seu bot"
            className={`${
              isDarkMode
                ? 'bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400'
                : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
            }`}
            maxLength={50}
          />
          <p
            className={`text-xs ${
              isDarkMode ? 'text-slate-400' : 'text-gray-500'
            }`}
          >
            {config.name.length}/50 caracteres
          </p>
        </div>

        {/* Personalidade */}
        <div className="space-y-2">
          <Label
            htmlFor="bot-personality"
            className={`flex items-center gap-2 text-sm font-medium ${
              isDarkMode ? 'text-slate-200' : 'text-gray-700'
            }`}
          >
            <Brain size={16} />
            Personalidade
          </Label>
          <Textarea
            id="bot-personality"
            value={config.personality}
            onChange={(e) => handleInputChange('personality', e.target.value)}
            placeholder="Descreva como seu bot deve se comportar e responder..."
            rows={4}
            className={`resize-none ${
              isDarkMode
                ? 'bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400'
                : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
            }`}
            maxLength={500}
          />
          <p
            className={`text-xs ${
              isDarkMode ? 'text-slate-400' : 'text-gray-500'
            }`}
          >
            {config.personality.length}/500 caracteres
          </p>
        </div>

        {/* Avatar */}
        <div className="space-y-2">
          <Label
            className={`flex items-center gap-2 text-sm font-medium ${
              isDarkMode ? 'text-slate-200' : 'text-gray-700'
            }`}
          >
            <Image size={16} />
            Avatar do Bot
          </Label>
          
          <div className="flex items-start gap-4">
            {/* Preview do Avatar */}
            <div
              className={`w-20 h-20 rounded-full border-2 border-dashed flex items-center justify-center overflow-hidden ${
                isDarkMode
                  ? 'border-slate-600 bg-slate-700'
                  : 'border-gray-300 bg-gray-50'
              }`}
            >
              {avatarPreview ? (
                <img
                  src={avatarPreview}
                  alt="Preview do avatar"
                  className="w-full h-full object-cover"
                />
              ) : (
                <User
                  size={24}
                  className={isDarkMode ? 'text-slate-400' : 'text-gray-400'}
                />
              )}
            </div>

            <div className="flex-1 space-y-3">
              {/* URL do Avatar */}
              <Input
                value={config.avatar}
                onChange={(e) => handleInputChange('avatar', e.target.value)}
                placeholder="URL da imagem ou cole uma imagem base64"
                className={`${
                  isDarkMode
                    ? 'bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400'
                    : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                }`}
              />

              {/* Upload de Arquivo */}
              <div className="flex items-center gap-2">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleAvatarUpload}
                  className="hidden"
                  id="avatar-upload"
                />
                <Label
                  htmlFor="avatar-upload"
                  className={`cursor-pointer flex items-center gap-2 px-3 py-2 text-sm rounded-md border transition-colors ${
                    isDarkMode
                      ? 'border-slate-600 text-slate-300 hover:bg-slate-700'
                      : 'border-gray-300 text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <Upload size={16} />
                  Fazer Upload
                </Label>
                <span
                  className={`text-xs ${
                    isDarkMode ? 'text-slate-400' : 'text-gray-500'
                  }`}
                >
                  ou cole uma URL
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Modelo OpenAI */}
        <div className="space-y-2">
          <Label
            className={`flex items-center gap-2 text-sm font-medium ${
              isDarkMode ? 'text-slate-200' : 'text-gray-700'
            }`}
          >
            <Brain size={16} />
            Modelo OpenAI
          </Label>
          <Select value={config.model} onValueChange={(value) => handleInputChange('model', value)}>
            <SelectTrigger
              className={`${
                isDarkMode
                  ? 'bg-slate-700 border-slate-600 text-slate-100'
                  : 'bg-white border-gray-300 text-gray-900'
              }`}
            >
              <SelectValue placeholder="Selecione um modelo" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gpt-4">GPT-4 (Recomendado)</SelectItem>
              <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
              <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
              <SelectItem value="gpt-4o">GPT-4o</SelectItem>
            </SelectContent>
          </Select>
          <p
            className={`text-xs ${
              isDarkMode ? 'text-slate-400' : 'text-gray-500'
            }`}
          >
            Escolha o modelo de IA que melhor atende às suas necessidades
          </p>
        </div>

        {/* Botão Salvar */}
        <div className="pt-4 border-t border-gray-200 dark:border-slate-600">
          <Button
            onClick={handleSave}
            disabled={!isFormValid || isSaving}
            className={`w-full flex items-center justify-center gap-2 py-3 ${
              isDarkMode
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            } ${
              saveStatus === 'success'
                ? 'bg-green-600 hover:bg-green-700'
                : saveStatus === 'error'
                ? 'bg-red-600 hover:bg-red-700'
                : ''
            }`}
          >
            {isSaving ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Salvando...
              </>
            ) : saveStatus === 'success' ? (
              <>
                <Save size={16} />
                Configurações Salvas!
              </>
            ) : saveStatus === 'error' ? (
              <>
                <Save size={16} />
                Erro ao Salvar
              </>
            ) : (
              <>
                <Save size={16} />
                Salvar Configurações
              </>
            )}
          </Button>

          {saveStatus === 'success' && (
            <p
              className={`text-center text-sm mt-2 ${
                isDarkMode ? 'text-green-400' : 'text-green-600'
              }`}
            >
              Suas configurações foram salvas com sucesso!
            </p>
          )}

          {saveStatus === 'error' && (
            <p
              className={`text-center text-sm mt-2 ${
                isDarkMode ? 'text-red-400' : 'text-red-600'
              }`}
            >
              Ocorreu um erro ao salvar. Tente novamente.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}