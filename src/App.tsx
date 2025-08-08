import './App.scss'

import { ThemeProvider } from "@/components/theme-provider"
import Footer from './components/Footer'
import ChatWindow from './components/ChatWindow'
import { MessagesProvider } from './components/context/messageContext'
import { UserDataProvider } from './components/context/userContext'
import { CompanionDataProvider } from './components/context/companionContext'
import { ConfigProvider } from './components/context/configContext'
import { AttitudeProvider } from './components/context/attitudeContext'

import { Toaster } from "@/components/ui/sonner"

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <ConfigProvider>
        <UserDataProvider>
          <CompanionDataProvider>
            <AttitudeProvider>
              <MessagesProvider>
                <div className='max-container'>
                  <ChatWindow />
                </div>
                <Toaster />
              </MessagesProvider>
            </AttitudeProvider>
          </CompanionDataProvider>
        </UserDataProvider>
      </ConfigProvider>
      <Footer />
    </ThemeProvider>
  )
}


export default App
