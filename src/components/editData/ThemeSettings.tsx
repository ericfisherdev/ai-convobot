import { Button } from "../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Label } from "../ui/label";
import { RadioGroup, RadioGroupItem } from "../ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Switch } from "../ui/switch";
import { Badge } from "../ui/badge";
import { Separator } from "../ui/separator";
import { Moon, Sun, Monitor, Palette, Type, Settings2, RotateCcw } from "lucide-react";
import { useThemeCustomization, type ThemeMode, type ColorScheme, type FontSize, type FontFamily } from "../context/themeContext";
import { toast } from "sonner";

export function ThemeSettings() {
  const { theme, setTheme, resetTheme } = useThemeCustomization();
  
  const handleReset = () => {
    resetTheme();
    toast.success("Theme settings reset to defaults");
  };

  const colorSchemes: { value: ColorScheme; label: string; color: string }[] = [
    { value: "default", label: "Default", color: "hsl(var(--primary))" },
    { value: "blue", label: "Ocean Blue", color: "hsl(214 100% 60%)" },
    { value: "green", label: "Forest Green", color: "hsl(142 76% 36%)" },
    { value: "purple", label: "Royal Purple", color: "hsl(263 70% 60%)" },
    { value: "orange", label: "Sunset Orange", color: "hsl(20 90% 55%)" },
    { value: "red", label: "Cherry Red", color: "hsl(0 84% 60%)" },
    { value: "pink", label: "Sakura Pink", color: "hsl(330 81% 60%)" },
    { value: "cyan", label: "Electric Cyan", color: "hsl(180 100% 50%)" },
  ];

  return (
    <div className="space-y-6 max-h-[80vh] overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings2 className="h-5 w-5" />
          <h2 className="text-lg font-semibold">Theme Customization</h2>
        </div>
        <Button variant="outline" size="sm" onClick={handleReset}>
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset
        </Button>
      </div>

      {/* Theme Mode */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Sun className="h-4 w-4" />
            <CardTitle className="text-sm">Appearance Mode</CardTitle>
          </div>
          <CardDescription>
            Choose your preferred theme appearance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RadioGroup
            value={theme.mode}
            onValueChange={(value) => setTheme({ mode: value as ThemeMode })}
            className="grid grid-cols-3 gap-4"
          >
            <div className="flex items-center space-x-2 border rounded-lg p-3 cursor-pointer hover:bg-muted/50">
              <RadioGroupItem value="light" id="light" />
              <div className="flex items-center gap-2">
                <Sun className="h-4 w-4" />
                <Label htmlFor="light" className="cursor-pointer">Light</Label>
              </div>
            </div>
            <div className="flex items-center space-x-2 border rounded-lg p-3 cursor-pointer hover:bg-muted/50">
              <RadioGroupItem value="dark" id="dark" />
              <div className="flex items-center gap-2">
                <Moon className="h-4 w-4" />
                <Label htmlFor="dark" className="cursor-pointer">Dark</Label>
              </div>
            </div>
            <div className="flex items-center space-x-2 border rounded-lg p-3 cursor-pointer hover:bg-muted/50">
              <RadioGroupItem value="system" id="system" />
              <div className="flex items-center gap-2">
                <Monitor className="h-4 w-4" />
                <Label htmlFor="system" className="cursor-pointer">System</Label>
              </div>
            </div>
          </RadioGroup>
        </CardContent>
      </Card>

      {/* Color Scheme */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Palette className="h-4 w-4" />
            <CardTitle className="text-sm">Color Scheme</CardTitle>
          </div>
          <CardDescription>
            Select your preferred color palette
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {colorSchemes.map((scheme) => (
              <div
                key={scheme.value}
                className={`
                  relative border rounded-lg p-3 cursor-pointer transition-all hover:border-primary/50
                  ${theme.colorScheme === scheme.value ? 'border-primary bg-primary/5' : ''}
                `}
                onClick={() => setTheme({ colorScheme: scheme.value })}
              >
                <div className="flex items-center gap-2">
                  <div 
                    className="w-4 h-4 rounded-full border-2 border-background shadow-sm"
                    style={{ backgroundColor: scheme.color }}
                  />
                  <span className="text-xs font-medium">{scheme.label}</span>
                </div>
                {theme.colorScheme === scheme.value && (
                  <Badge variant="secondary" className="absolute -top-1 -right-1 text-xs px-1">
                    ✓
                  </Badge>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Typography */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Type className="h-4 w-4" />
            <CardTitle className="text-sm">Typography</CardTitle>
          </div>
          <CardDescription>
            Customize font size and family
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="fontSize">Font Size</Label>
            <Select value={theme.fontSize} onValueChange={(value) => setTheme({ fontSize: value as FontSize })}>
              <SelectTrigger>
                <SelectValue placeholder="Select font size" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="xs">Extra Small</SelectItem>
                <SelectItem value="sm">Small</SelectItem>
                <SelectItem value="md">Medium</SelectItem>
                <SelectItem value="lg">Large</SelectItem>
                <SelectItem value="xl">Extra Large</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="fontFamily">Font Family</Label>
            <Select value={theme.fontFamily} onValueChange={(value) => setTheme({ fontFamily: value as FontFamily })}>
              <SelectTrigger>
                <SelectValue placeholder="Select font family" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="system">System Default</SelectItem>
                <SelectItem value="sans">Sans Serif</SelectItem>
                <SelectItem value="serif">Serif</SelectItem>
                <SelectItem value="mono">Monospace</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Accessibility */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Accessibility & Performance</CardTitle>
          <CardDescription>
            Options to improve usability and performance
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="animations">Enable Animations</Label>
              <p className="text-xs text-muted-foreground">
                Smooth transitions and micro-interactions
              </p>
            </div>
            <Switch
              id="animations"
              checked={theme.animations}
              onCheckedChange={(checked) => setTheme({ animations: checked })}
            />
          </div>
          
          <Separator />
          
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="compactMode">Compact Mode</Label>
              <p className="text-xs text-muted-foreground">
                Reduce spacing for more content visibility
              </p>
            </div>
            <Switch
              id="compactMode"
              checked={theme.compactMode}
              onCheckedChange={(checked) => setTheme({ compactMode: checked })}
            />
          </div>
          
          <Separator />
          
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="highContrast">High Contrast</Label>
              <p className="text-xs text-muted-foreground">
                Enhanced contrast for better readability
              </p>
            </div>
            <Switch
              id="highContrast"
              checked={theme.highContrast}
              onCheckedChange={(checked) => setTheme({ highContrast: checked })}
            />
          </div>
        </CardContent>
      </Card>

      {/* Preview */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Preview</CardTitle>
          <CardDescription>
            See how your theme looks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 p-4 rounded-lg border bg-card">
            <div className="flex justify-between items-center">
              <span className="font-medium">Sample Message</span>
              <span className="text-xs text-muted-foreground">just now</span>
            </div>
            <div className="bg-primary text-primary-foreground p-3 rounded-lg max-w-xs">
              Hello! This is how messages will look with your current theme settings.
            </div>
            <div className="bg-secondary text-secondary-foreground p-3 rounded-lg max-w-xs ml-auto">
              And this is how AI responses will appear.
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}