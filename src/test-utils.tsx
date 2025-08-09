import React, { createContext, ReactNode } from 'react'

// Mock contexts
export const MockMessageContext = createContext({
  messages: [],
  loading: false,
  error: null,
  addMessage: () => {},
  deleteMessage: () => {},
  editMessage: () => {},
  refreshMessages: () => Promise.resolve(),
})

export const MockUserDataContext = createContext({
  userData: { name: 'Test User', persona: 'Test persona' },
  loading: false,
  error: null,
  updateUserData: () => Promise.resolve(),
})

export const MockCompanionDataContext = createContext({
  companionData: { 
    name: 'Test Companion', 
    persona: 'Test companion persona',
    example_dialogue: 'Hello!',
    first_message: 'Welcome!',
    long_term_mem: 100,
    short_term_mem: 50,
    roleplay: true,
    dialogue_tuning: 1,
    avatar_path: '/avatar.jpg'
  },
  loading: false,
  error: null,
  updateCompanionData: () => Promise.resolve(),
})

export const MockConfigContext = createContext({
  config: {
    device: 'CPU',
    llm_model_path: '/model.gguf',
    gpu_layers: 0,
    prompt_template: 'Default',
    context_window_size: 2048,
    max_response_tokens: 512,
    enable_dynamic_context: true,
    vram_limit_gb: 4
  },
  loading: false,
  error: null,
  updateConfig: () => Promise.resolve(),
})

export const MockAttitudeContext = createContext({
  attitudes: [],
  loading: false,
  error: null,
  updateAttitude: () => Promise.resolve(),
})

export const MockThemeContext = createContext({
  theme: 'light' as const,
  setTheme: () => {},
})

// Mock providers
export const MockMessageProvider: React.FC<{ children: ReactNode }> = ({ children }) => (
  <MockMessageContext.Provider value={MockMessageContext._defaultValue}>
    {children}
  </MockMessageContext.Provider>
)

export const MockUserDataProvider: React.FC<{ children: ReactNode }> = ({ children }) => (
  <MockUserDataContext.Provider value={MockUserDataContext._defaultValue}>
    {children}
  </MockUserDataContext.Provider>
)

export const MockCompanionDataProvider: React.FC<{ children: ReactNode }> = ({ children }) => (
  <MockCompanionDataContext.Provider value={MockCompanionDataContext._defaultValue}>
    {children}
  </MockCompanionDataContext.Provider>
)

export const MockConfigProvider: React.FC<{ children: ReactNode }> = ({ children }) => (
  <MockConfigContext.Provider value={MockConfigContext._defaultValue}>
    {children}
  </MockConfigContext.Provider>
)

export const MockAttitudeProvider: React.FC<{ children: ReactNode }> = ({ children }) => (
  <MockAttitudeContext.Provider value={MockAttitudeContext._defaultValue}>
    {children}
  </MockAttitudeContext.Provider>
)

export const MockThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => (
  <MockThemeContext.Provider value={MockThemeContext._defaultValue}>
    <div className="mock-theme-provider">
      {children}
    </div>
  </MockThemeContext.Provider>
)

// All providers wrapper
export const TestProviders: React.FC<{ children: ReactNode }> = ({ children }) => (
  <MockThemeProvider>
    <MockMessageProvider>
      <MockUserDataProvider>
        <MockCompanionDataProvider>
          <MockConfigProvider>
            <MockAttitudeProvider>
              {children}
            </MockAttitudeProvider>
          </MockConfigProvider>
        </MockCompanionDataProvider>
      </MockUserDataProvider>
    </MockMessageProvider>
  </MockThemeProvider>
)