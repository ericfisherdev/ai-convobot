import '@testing-library/jest-dom'

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// jsdom never fires a plain `Image()`'s `load` event (there is no real
// network), which leaves every `@radix-ui/react-avatar` `AvatarImage`
// (it probes loading status with `new window.Image()`) permanently
// showing its `AvatarFallback` in tests. Firing `onload` as soon as `src`
// is set lets tests assert on the rendered `<img>` instead.
class MockImage {
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  private _src = ''
  get src() {
    return this._src
  }
  set src(value: string) {
    this._src = value
    if (value) {
      queueMicrotask(() => this.onload?.())
    }
  }
}
Object.defineProperty(window, 'Image', {
  writable: true,
  value: MockImage,
})
