import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { AttitudeManager } from '../attitude/AttitudeManager'
import { AttitudeProvider } from '../context/attitudeContext'

const MockAttitudeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AttitudeProvider>
    {children}
  </AttitudeProvider>
)

describe('AttitudeManager Component', () => {
  beforeEach(() => {
    // Mock fetch for API calls
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          attraction: 25,
          trust: 75,
          fear: 10,
          anger: 5,
          joy: 60,
          sorrow: 15,
          disgust: 8,
          surprise: 30,
          curiosity: 85,
          respect: 70,
          suspicion: 20,
          gratitude: 40,
          jealousy: 12,
          empathy: 80
        }),
      })
    ) as vi.Mock
  })

  it('renders attitude manager', () => {
    render(
      <MockAttitudeProvider>
        <AttitudeManager />
      </MockAttitudeProvider>
    )
    
    // Check if the attitude manager container is present
    expect(screen.getByTestId('attitude-manager')).toBeInTheDocument()
  })

  it('displays attitude dimensions', async () => {
    render(
      <MockAttitudeProvider>
        <AttitudeManager />
      </MockAttitudeProvider>
    )
    
    // Wait for attitude data to load and check for key dimensions
    expect(screen.getByText(/Trust/i)).toBeInTheDocument()
    expect(screen.getByText(/Joy/i)).toBeInTheDocument()
    expect(screen.getByText(/Curiosity/i)).toBeInTheDocument()
  })

  it('shows attitude values as progress bars or indicators', async () => {
    render(
      <MockAttitudeProvider>
        <AttitudeManager />
      </MockAttitudeProvider>
    )
    
    // Check for progress indicators
    const progressBars = screen.getAllByRole('progressbar')
    expect(progressBars.length).toBeGreaterThan(0)
  })
})