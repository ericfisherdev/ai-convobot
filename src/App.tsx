import './App.scss'

import { ThemeProvider } from "@/components/theme-provider"
import { ThemeCustomizationProvider } from './components/context/themeContext'
import Footer from './components/Footer'
import ChatWindow from './components/ChatWindow'
import { PWAInstallPrompt } from './components/mobile/PWAInstallPrompt'
import { MessagesProvider } from './components/context/messageContext'
import { UserDataProvider } from './components/context/userContext'
import { CompanionDataProvider } from './components/context/companionContext'
import { ConfigProvider } from './components/context/configContext'
import { AttitudeProvider } from './components/context/attitudeContext'
import { useMobile } from './hooks/useMobile'

import { Toaster } from "@/components/ui/sonner"

function App() {
  const { isMobile, isStandalone } = useMobile();
  
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <ThemeCustomizationProvider>
        <ConfigProvider>
          <UserDataProvider>
            <CompanionDataProvider>
              <AttitudeProvider>
                <MessagesProvider>
                  <div className='max-container'>
                    <ChatWindow />
                  </div>
                  <Toaster />
                  <PWAInstallPrompt />
                </MessagesProvider>
              </AttitudeProvider>
            </CompanionDataProvider>
          </UserDataProvider>
        </ConfigProvider>
        {/* Only show footer on desktop or non-PWA */}
        {(!isMobile || !isStandalone) && <Footer />}
      </ThemeCustomizationProvider>
    </ThemeProvider>
  )
}


export default App
