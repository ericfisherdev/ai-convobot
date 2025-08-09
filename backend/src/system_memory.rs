use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMemoryInfo {
    pub total_ram_gb: f32,
    pub available_ram_gb: f32,
    pub used_ram_gb: f32,
    pub utilization_percent: f32,
    pub platform: String,
    pub detection_method: String,
}

#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub available_for_context_gb: f32,
    pub safety_margin_gb: f32,
    pub recommended_usage_gb: f32,
    pub allocation_strategy: MemoryStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryStrategy {
    VramOnly,      // <8GB system RAM or high utilization
    Conservative,  // 8-16GB system RAM: use 25% of available
    Balanced,      // 16-32GB system RAM: use 50% of available
    Aggressive,    // 32GB+ system RAM: use up to 75% of available
}

pub struct SystemMemoryDetector {
    safety_margin_gb: f32,
    max_usage_gb: f32,
}

impl Default for SystemMemoryDetector {
    fn default() -> Self {
        Self {
            safety_margin_gb: 2.0, // Keep 2GB free for system
            max_usage_gb: 8.0,     // Default max 8GB for context
        }
    }
}

impl SystemMemoryDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_safety_margin(mut self, margin_gb: f32) -> Self {
        self.safety_margin_gb = margin_gb.max(0.5); // Minimum 0.5GB safety margin
        self
    }

    pub fn with_max_usage(mut self, max_gb: f32) -> Self {
        self.max_usage_gb = max_gb.max(1.0); // Minimum 1GB max usage
        self
    }

    /// Detect system memory information cross-platform
    pub fn detect_system_memory(&self) -> Result<SystemMemoryInfo, Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        return self.detect_linux_memory();

        #[cfg(target_os = "windows")]
        return self.detect_windows_memory();

        #[cfg(target_os = "macos")]
        return self.detect_macos_memory();

        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            // Fallback for unsupported platforms
            Ok(SystemMemoryInfo {
                total_ram_gb: 8.0, // Conservative fallback
                available_ram_gb: 4.0,
                used_ram_gb: 4.0,
                utilization_percent: 50.0,
                platform: "unsupported".to_string(),
                detection_method: "fallback".to_string(),
            })
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_memory(&self) -> Result<SystemMemoryInfo, Box<dyn std::error::Error>> {
        use std::fs;

        // Read /proc/meminfo for detailed memory information
        let meminfo = fs::read_to_string("/proc/meminfo")?;
        
        let mut total_kb = 0u64;
        let mut available_kb = 0u64;
        let mut free_kb = 0u64;
        let mut buffers_kb = 0u64;
        let mut cached_kb = 0u64;

        for line in meminfo.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let value = parts[1].parse::<u64>().unwrap_or(0);
                match parts[0] {
                    "MemTotal:" => total_kb = value,
                    "MemAvailable:" => available_kb = value,
                    "MemFree:" => free_kb = value,
                    "Buffers:" => buffers_kb = value,
                    "Cached:" => cached_kb = value,
                    _ => {}
                }
            }
        }

        // If MemAvailable is not available, estimate it
        if available_kb == 0 {
            available_kb = free_kb + buffers_kb + cached_kb;
        }

        let total_ram_gb = total_kb as f32 / 1024.0 / 1024.0;
        let available_ram_gb = available_kb as f32 / 1024.0 / 1024.0;
        let used_ram_gb = total_ram_gb - available_ram_gb;
        let utilization_percent = (used_ram_gb / total_ram_gb) * 100.0;

        Ok(SystemMemoryInfo {
            total_ram_gb,
            available_ram_gb,
            used_ram_gb,
            utilization_percent,
            platform: "linux".to_string(),
            detection_method: "proc_meminfo".to_string(),
        })
    }

    #[cfg(target_os = "windows")]
    fn detect_windows_memory(&self) -> Result<SystemMemoryInfo, Box<dyn std::error::Error>> {
        // Use PowerShell to get memory information
        let output = Command::new("powershell")
            .args(&[
                "-Command",
                "Get-CimInstance -ClassName Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory | ConvertTo-Json"
            ])
            .output()?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            
            // Try to parse JSON output
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                let total_kb = json["TotalVisibleMemorySize"].as_u64().unwrap_or(8 * 1024 * 1024);
                let free_kb = json["FreePhysicalMemory"].as_u64().unwrap_or(4 * 1024 * 1024);
                
                let total_ram_gb = total_kb as f32 / 1024.0 / 1024.0;
                let available_ram_gb = free_kb as f32 / 1024.0 / 1024.0;
                let used_ram_gb = total_ram_gb - available_ram_gb;
                let utilization_percent = (used_ram_gb / total_ram_gb) * 100.0;

                return Ok(SystemMemoryInfo {
                    total_ram_gb,
                    available_ram_gb,
                    used_ram_gb,
                    utilization_percent,
                    platform: "windows".to_string(),
                    detection_method: "powershell_cim".to_string(),
                });
            }
        }

        // Fallback: try wmic command
        let output = Command::new("wmic")
            .args(&["OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/format:csv"])
            .output()?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines().skip(1) { // Skip header
                let fields: Vec<&str> = line.split(',').collect();
                if fields.len() >= 3 {
                    if let (Ok(free_kb), Ok(total_kb)) = (
                        fields[1].trim().parse::<u64>(),
                        fields[2].trim().parse::<u64>(),
                    ) {
                        let total_ram_gb = total_kb as f32 / 1024.0 / 1024.0;
                        let available_ram_gb = free_kb as f32 / 1024.0 / 1024.0;
                        let used_ram_gb = total_ram_gb - available_ram_gb;
                        let utilization_percent = (used_ram_gb / total_ram_gb) * 100.0;

                        return Ok(SystemMemoryInfo {
                            total_ram_gb,
                            available_ram_gb,
                            used_ram_gb,
                            utilization_percent,
                            platform: "windows".to_string(),
                            detection_method: "wmic".to_string(),
                        });
                    }
                }
            }
        }

        // Final fallback for Windows
        Ok(SystemMemoryInfo {
            total_ram_gb: 16.0, // Common Windows system
            available_ram_gb: 8.0,
            used_ram_gb: 8.0,
            utilization_percent: 50.0,
            platform: "windows".to_string(),
            detection_method: "fallback".to_string(),
        })
    }

    #[cfg(target_os = "macos")]
    fn detect_macos_memory(&self) -> Result<SystemMemoryInfo, Box<dyn std::error::Error>> {
        // Use vm_stat and system_profiler for macOS memory detection
        let vm_stat_output = Command::new("vm_stat").output()?;
        let sysctl_output = Command::new("sysctl")
            .args(&["hw.memsize"])
            .output()?;

        let mut total_ram_gb = 16.0; // Fallback
        let mut free_pages = 0u64;
        let page_size = 4096u64; // Standard page size

        // Parse total memory from sysctl
        if sysctl_output.status.success() {
            let stdout = String::from_utf8_lossy(&sysctl_output.stdout);
            for line in stdout.lines() {
                if line.starts_with("hw.memsize:") {
                    if let Some(value_str) = line.split(": ").nth(1) {
                        if let Ok(bytes) = value_str.trim().parse::<u64>() {
                            total_ram_gb = bytes as f32 / 1024.0 / 1024.0 / 1024.0;
                        }
                    }
                }
            }
        }

        // Parse free pages from vm_stat
        if vm_stat_output.status.success() {
            let stdout = String::from_utf8_lossy(&vm_stat_output.stdout);
            for line in stdout.lines() {
                if line.starts_with("Pages free:") {
                    if let Some(pages_str) = line.split(": ").nth(1) {
                        let pages_str = pages_str.trim_end_matches('.');
                        if let Ok(pages) = pages_str.parse::<u64>() {
                            free_pages = pages;
                        }
                    }
                }
            }
        }

        let available_ram_gb = (free_pages * page_size) as f32 / 1024.0 / 1024.0 / 1024.0;
        let used_ram_gb = total_ram_gb - available_ram_gb;
        let utilization_percent = (used_ram_gb / total_ram_gb) * 100.0;

        Ok(SystemMemoryInfo {
            total_ram_gb,
            available_ram_gb: available_ram_gb.max(1.0), // Ensure at least 1GB reported available
            used_ram_gb,
            utilization_percent,
            platform: "macos".to_string(),
            detection_method: "vm_stat_sysctl".to_string(),
        })
    }

    /// Calculate optimal memory allocation for context expansion
    pub fn calculate_memory_allocation(
        &self,
        memory_info: &SystemMemoryInfo,
    ) -> MemoryAllocation {
        let available_after_safety = (memory_info.available_ram_gb - self.safety_margin_gb).max(0.0);
        
        let strategy = self.determine_strategy(memory_info);
        
        let recommended_usage_gb = match strategy {
            MemoryStrategy::VramOnly => 0.0,
            MemoryStrategy::Conservative => (available_after_safety * 0.25).min(self.max_usage_gb),
            MemoryStrategy::Balanced => (available_after_safety * 0.50).min(self.max_usage_gb),
            MemoryStrategy::Aggressive => (available_after_safety * 0.75).min(self.max_usage_gb),
        };

        MemoryAllocation {
            available_for_context_gb: available_after_safety,
            safety_margin_gb: self.safety_margin_gb,
            recommended_usage_gb: recommended_usage_gb.max(0.0),
            allocation_strategy: strategy,
        }
    }

    /// Determine memory allocation strategy based on system characteristics
    fn determine_strategy(&self, memory_info: &SystemMemoryInfo) -> MemoryStrategy {
        // Check if system is under memory pressure
        if memory_info.utilization_percent > 85.0 || memory_info.available_ram_gb < 4.0 {
            return MemoryStrategy::VramOnly;
        }

        // Determine strategy based on total RAM
        match memory_info.total_ram_gb as u32 {
            0..=7 => MemoryStrategy::VramOnly,
            8..=15 => MemoryStrategy::Conservative,
            16..=31 => MemoryStrategy::Balanced,
            _ => MemoryStrategy::Aggressive,
        }
    }

    /// Check if system memory is under pressure
    pub fn is_memory_pressure(&self, memory_info: &SystemMemoryInfo) -> bool {
        memory_info.utilization_percent > 85.0 || memory_info.available_ram_gb < self.safety_margin_gb
    }

    /// Get memory status summary for logging
    pub fn get_memory_summary(&self, memory_info: &SystemMemoryInfo) -> String {
        let allocation = self.calculate_memory_allocation(memory_info);
        format!(
            "System Memory: {:.1}GB total, {:.1}GB available ({:.1}% used) - Strategy: {:?}, Context allocation: {:.1}GB",
            memory_info.total_ram_gb,
            memory_info.available_ram_gb,
            memory_info.utilization_percent,
            allocation.allocation_strategy,
            allocation.recommended_usage_gb
        )
    }
}

impl std::fmt::Display for SystemMemoryInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "System RAM: {:.1}GB total, {:.1}GB available ({:.1}% used) [{}]",
            self.total_ram_gb, self.available_ram_gb, self.utilization_percent, self.platform
        )
    }
}

impl std::fmt::Display for MemoryAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Memory allocation: {:.1}GB available for context ({:?}), recommended: {:.1}GB",
            self.available_for_context_gb, self.allocation_strategy, self.recommended_usage_gb
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_strategy_determination() {
        let detector = SystemMemoryDetector::new();

        // Test VramOnly conditions
        let low_ram = SystemMemoryInfo {
            total_ram_gb: 4.0,
            available_ram_gb: 1.0,
            used_ram_gb: 3.0,
            utilization_percent: 75.0,
            platform: "test".to_string(),
            detection_method: "test".to_string(),
        };
        assert!(matches!(
            detector.determine_strategy(&low_ram),
            MemoryStrategy::VramOnly
        ));

        // Test Conservative strategy
        let medium_ram = SystemMemoryInfo {
            total_ram_gb: 12.0,
            available_ram_gb: 6.0,
            used_ram_gb: 6.0,
            utilization_percent: 50.0,
            platform: "test".to_string(),
            detection_method: "test".to_string(),
        };
        assert!(matches!(
            detector.determine_strategy(&medium_ram),
            MemoryStrategy::Conservative
        ));

        // Test Balanced strategy
        let high_ram = SystemMemoryInfo {
            total_ram_gb: 24.0,
            available_ram_gb: 12.0,
            used_ram_gb: 12.0,
            utilization_percent: 50.0,
            platform: "test".to_string(),
            detection_method: "test".to_string(),
        };
        assert!(matches!(
            detector.determine_strategy(&high_ram),
            MemoryStrategy::Balanced
        ));

        // Test Aggressive strategy
        let very_high_ram = SystemMemoryInfo {
            total_ram_gb: 64.0,
            available_ram_gb: 32.0,
            used_ram_gb: 32.0,
            utilization_percent: 50.0,
            platform: "test".to_string(),
            detection_method: "test".to_string(),
        };
        assert!(matches!(
            detector.determine_strategy(&very_high_ram),
            MemoryStrategy::Aggressive
        ));
    }

    #[test]
    fn test_memory_allocation_calculation() {
        let detector = SystemMemoryDetector::new().with_safety_margin(2.0);

        let memory_info = SystemMemoryInfo {
            total_ram_gb: 16.0,
            available_ram_gb: 10.0,
            used_ram_gb: 6.0,
            utilization_percent: 37.5,
            platform: "test".to_string(),
            detection_method: "test".to_string(),
        };

        let allocation = detector.calculate_memory_allocation(&memory_info);

        assert_eq!(allocation.available_for_context_gb, 8.0); // 10 - 2 safety margin
        assert_eq!(allocation.safety_margin_gb, 2.0);
        // Should be Balanced strategy (16GB RAM), so 50% of available = 4.0GB
        assert_eq!(allocation.recommended_usage_gb, 4.0);
        assert!(matches!(allocation.allocation_strategy, MemoryStrategy::Balanced));
    }

    #[test]
    fn test_memory_pressure_detection() {
        let detector = SystemMemoryDetector::new();

        let high_pressure = SystemMemoryInfo {
            total_ram_gb: 8.0,
            available_ram_gb: 1.0,
            used_ram_gb: 7.0,
            utilization_percent: 87.5,
            platform: "test".to_string(),
            detection_method: "test".to_string(),
        };
        assert!(detector.is_memory_pressure(&high_pressure));

        let normal_pressure = SystemMemoryInfo {
            total_ram_gb: 16.0,
            available_ram_gb: 8.0,
            used_ram_gb: 8.0,
            utilization_percent: 50.0,
            platform: "test".to_string(),
            detection_method: "test".to_string(),
        };
        assert!(!detector.is_memory_pressure(&normal_pressure));
    }
}