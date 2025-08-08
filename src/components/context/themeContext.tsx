import { createContext, useContext, useEffect, useState } from "react";

export type ThemeMode = "dark" | "light" | "system";

export type ColorScheme = 
  | "default"
  | "blue" 
  | "green"
  | "purple"
  | "orange"
  | "red"
  | "pink"
  | "cyan";

export type FontSize = "xs" | "sm" | "md" | "lg" | "xl";
export type FontFamily = "system" | "mono" | "serif" | "sans";

export interface ThemeCustomization {
  mode: ThemeMode;
  colorScheme: ColorScheme;
  fontSize: FontSize;
  fontFamily: FontFamily;
  animations: boolean;
  compactMode: boolean;
  highContrast: boolean;
}

interface ThemeContextValue {
  theme: ThemeCustomization;
  setTheme: (theme: Partial<ThemeCustomization>) => void;
  resetTheme: () => void;
}

const defaultTheme: ThemeCustomization = {
  mode: "dark",
  colorScheme: "default",
  fontSize: "md",
  fontFamily: "system",
  animations: true,
  compactMode: false,
  highContrast: false,
};

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

export function ThemeCustomizationProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<ThemeCustomization>(() => {
    const stored = localStorage.getItem('ai-companion-theme');
    return stored ? { ...defaultTheme, ...JSON.parse(stored) } : defaultTheme;
  });

  const setTheme = (updates: Partial<ThemeCustomization>) => {
    const newTheme = { ...theme, ...updates };
    setThemeState(newTheme);
    localStorage.setItem('ai-companion-theme', JSON.stringify(newTheme));
  };

  const resetTheme = () => {
    setThemeState(defaultTheme);
    localStorage.setItem('ai-companion-theme', JSON.stringify(defaultTheme));
  };

  useEffect(() => {
    const root = document.documentElement;
    
    // Clear previous classes
    root.classList.remove('light', 'dark', 'system');
    root.classList.remove('scheme-blue', 'scheme-green', 'scheme-purple', 'scheme-orange', 'scheme-red', 'scheme-pink', 'scheme-cyan');
    root.classList.remove('font-xs', 'font-sm', 'font-md', 'font-lg', 'font-xl');
    root.classList.remove('font-system', 'font-mono', 'font-serif', 'font-sans');
    root.classList.remove('animations-disabled', 'compact-mode', 'high-contrast');
    
    // Apply theme mode
    if (theme.mode === "system") {
      const systemTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      root.classList.add(systemTheme);
    } else {
      root.classList.add(theme.mode);
    }
    
    // Apply color scheme
    if (theme.colorScheme !== 'default') {
      root.classList.add(`scheme-${theme.colorScheme}`);
    }
    
    // Apply font settings
    root.classList.add(`font-${theme.fontSize}`);
    root.classList.add(`font-${theme.fontFamily}`);
    
    // Apply accessibility options
    if (!theme.animations) {
      root.classList.add('animations-disabled');
    }
    if (theme.compactMode) {
      root.classList.add('compact-mode');
    }
    if (theme.highContrast) {
      root.classList.add('high-contrast');
    }
    
    // Update CSS custom properties for dynamic theming
    if (theme.colorScheme !== 'default') {
      updateColorScheme(theme.colorScheme);
    }
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme, resetTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useThemeCustomization() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useThemeCustomization must be used within a ThemeCustomizationProvider');
  }
  return context;
}

function updateColorScheme(scheme: ColorScheme) {
  const root = document.documentElement;
  
  const colorSchemes: Record<Exclude<ColorScheme, 'default'>, Record<string, string>> = {
    blue: {
      primary: '214 100% 60%',
      primaryForeground: '0 0% 98%',
      secondary: '214 20% 14%',
      accent: '214 25% 18%',
    },
    green: {
      primary: '142 76% 36%',
      primaryForeground: '0 0% 98%',
      secondary: '142 13% 15%', 
      accent: '142 16% 20%',
    },
    purple: {
      primary: '263 70% 60%',
      primaryForeground: '0 0% 98%',
      secondary: '263 15% 15%',
      accent: '263 20% 20%',
    },
    orange: {
      primary: '20 90% 55%',
      primaryForeground: '0 0% 98%',
      secondary: '20 15% 15%',
      accent: '20 20% 20%',
    },
    red: {
      primary: '0 84% 60%',
      primaryForeground: '0 0% 98%',
      secondary: '0 15% 15%',
      accent: '0 20% 20%',
    },
    pink: {
      primary: '330 81% 60%',
      primaryForeground: '0 0% 98%',
      secondary: '330 15% 15%',
      accent: '330 20% 20%',
    },
    cyan: {
      primary: '180 100% 50%',
      primaryForeground: '0 0% 98%',
      secondary: '180 15% 15%',
      accent: '180 20% 20%',
    },
  };
  
  if (scheme !== 'default') {
    const colors = colorSchemes[scheme];
    if (colors) {
      Object.entries(colors).forEach(([key, value]) => {
        root.style.setProperty(`--${key}`, value as string);
      });
    }
  }
}