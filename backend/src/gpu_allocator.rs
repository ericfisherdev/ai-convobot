use crate::database::Device;
use crate::model_metadata::ModelFacts;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryInfo {
    pub total_vram_mb: u64,
    pub available_vram_mb: u64,
    pub used_vram_mb: u64,
    pub utilization_percent: f32,
    pub device_name: String,
    pub driver_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAllocation {
    pub gpu_layers: usize,
    pub cpu_layers: usize,
    pub total_layers: usize,
    pub estimated_vram_usage_mb: u64,
    pub allocation_strategy: AllocationStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    MaxGpu,
    Balanced,
    Conservative,
    CpuFallback,
    Aggressive, // New strategy for systems with 3.5GB+ VRAM
}

pub struct GpuAllocator {
    safety_margin_percent: f32,
    min_free_vram_mb: u64,
}

impl Default for GpuAllocator {
    fn default() -> Self {
        Self {
            safety_margin_percent: 0.8, // Use only 80% of available VRAM
            min_free_vram_mb: 512,      // Keep at least 512MB free
        }
    }
}

impl GpuAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_safety_margin(mut self, margin_percent: f32) -> Self {
        self.safety_margin_percent = margin_percent.clamp(0.1, 1.0);
        self
    }

    pub fn with_min_free_vram(mut self, min_vram_mb: u64) -> Self {
        self.min_free_vram_mb = min_vram_mb;
        self
    }

    /// Detect GPU memory information based on device type
    pub fn detect_gpu_memory(
        &self,
        device: &Device,
    ) -> Result<GpuMemoryInfo, Box<dyn std::error::Error>> {
        match device {
            Device::GPU => self.detect_cuda_memory(),
            Device::Metal => self.detect_metal_memory(),
            Device::CPU => Ok(GpuMemoryInfo {
                total_vram_mb: 0,
                available_vram_mb: 0,
                used_vram_mb: 0,
                utilization_percent: 0.0,
                device_name: "CPU".to_string(),
                driver_version: "N/A".to_string(),
            }),
        }
    }

    /// Detect CUDA GPU memory using nvidia-smi
    fn detect_cuda_memory(&self) -> Result<GpuMemoryInfo, Box<dyn std::error::Error>> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,name,driver_version",
                "--format=csv,noheader,nounits"
            ])
            .output();

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout.lines().next() {
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 6 {
                        let total_vram_mb: u64 = parts[0].trim().parse().unwrap_or(0);
                        let used_vram_mb: u64 = parts[1].trim().parse().unwrap_or(0);
                        let available_vram_mb = total_vram_mb.saturating_sub(used_vram_mb);
                        let utilization_percent: f32 = parts[3].trim().parse().unwrap_or(0.0);
                        let device_name = parts[4].trim().to_string();
                        let driver_version = parts[5].trim().to_string();

                        return Ok(GpuMemoryInfo {
                            total_vram_mb,
                            available_vram_mb,
                            used_vram_mb,
                            utilization_percent,
                            device_name,
                            driver_version,
                        });
                    }
                }
                Err("Failed to parse nvidia-smi output".into())
            }
            Err(_) => {
                // Fallback: estimate based on common GPU configurations
                println!("Warning: nvidia-smi not available, using fallback estimation");
                self.estimate_gpu_memory_fallback()
            }
        }
    }

    /// Detect Metal GPU memory (macOS)
    fn detect_metal_memory(&self) -> Result<GpuMemoryInfo, Box<dyn std::error::Error>> {
        let output = Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output();

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);

                // Parse system_profiler output for GPU memory
                // This is a simplified version - real implementation would parse JSON
                if stdout.contains("VRAM") {
                    // Extract VRAM info from system profiler output
                    let estimated_vram = 8192; // Default estimation for modern Macs
                    Ok(GpuMemoryInfo {
                        total_vram_mb: estimated_vram,
                        available_vram_mb: estimated_vram * 70 / 100, // Assume 70% available
                        used_vram_mb: estimated_vram * 30 / 100,
                        utilization_percent: 30.0,
                        device_name: "Apple GPU".to_string(),
                        driver_version: "macOS".to_string(),
                    })
                } else {
                    self.estimate_gpu_memory_fallback()
                }
            }
            Err(_) => {
                println!("Warning: system_profiler not available, using fallback estimation");
                self.estimate_gpu_memory_fallback()
            }
        }
    }

    /// Fallback GPU memory estimation when detection tools are unavailable
    fn estimate_gpu_memory_fallback(&self) -> Result<GpuMemoryInfo, Box<dyn std::error::Error>> {
        // Conservative estimation for unknown GPUs
        let estimated_vram = 4096; // 4GB default
        Ok(GpuMemoryInfo {
            total_vram_mb: estimated_vram,
            available_vram_mb: estimated_vram * 60 / 100, // Conservative 60% available
            used_vram_mb: estimated_vram * 40 / 100,
            utilization_percent: 40.0,
            device_name: "Unknown GPU".to_string(),
            driver_version: "Unknown".to_string(),
        })
    }

    /// Calculate optimal GPU layer allocation.
    ///
    /// The per-layer VRAM estimate below is derived from `model_size_mb`
    /// and `total_layers` directly — the file size on disk already
    /// reflects whatever quantization produced it, so no separate
    /// quantization guess is needed. `model_path` is kept for the log line.
    pub fn calculate_optimal_layers_v2(
        &self,
        gpu_info: &GpuMemoryInfo,
        model_path: &str,
        model_size_mb: u64,
        total_layers: usize,
        vram_limit_gb: Option<f32>,
    ) -> LayerAllocation {
        println!(
            "📦 Sizing '{}': {} MB across {} layers",
            model_path, model_size_mb, total_layers
        );

        // Apply VRAM limit if specified
        let effective_available_vram = if let Some(limit_gb) = vram_limit_gb {
            let limit_mb = (limit_gb * 1024.0) as u64;
            std::cmp::min(gpu_info.available_vram_mb, limit_mb)
        } else {
            gpu_info.available_vram_mb
        };

        // Use aggressive allocation for systems with 3.5GB+ VRAM
        let (safety_margin, min_free_mb) = if gpu_info.total_vram_mb >= 3500 {
            (0.90, 256u64) // Use 90% of VRAM, keep only 256MB free for 3.5GB+ systems
        } else if gpu_info.total_vram_mb >= 2048 {
            (0.85, 384u64) // Use 85% for 2GB+ systems
        } else {
            (self.safety_margin_percent, self.min_free_vram_mb) // Use configured values for smaller GPUs
        };

        // Apply safety margin
        let safe_vram_mb = ((effective_available_vram as f32) * safety_margin) as u64;
        let usable_vram_mb = safe_vram_mb.saturating_sub(min_free_mb);

        // VRAM estimation per layer: the model's on-disk size spread evenly
        // across its layers, plus a reservation for KV cache and
        // architecture overhead.
        let kv_cache_overhead_mb = 200; // Reserve for KV cache and context
        let arch_overhead_mb = 100; // Reserve for model architecture overhead

        let vram_per_layer_mb = (model_size_mb as f32 / total_layers.max(1) as f32).ceil() as u64;
        let reserved_vram = kv_cache_overhead_mb + arch_overhead_mb;
        let layers_vram_budget = usable_vram_mb.saturating_sub(reserved_vram);

        // Calculate maximum layers that fit in VRAM
        let max_gpu_layers = if vram_per_layer_mb > 0 && layers_vram_budget > 0 {
            std::cmp::min(
                (layers_vram_budget / vram_per_layer_mb) as usize,
                total_layers,
            )
        } else {
            0
        };

        // Determine allocation strategy with new aggressive option
        let (gpu_layers, strategy) = if max_gpu_layers == 0 {
            (0, AllocationStrategy::CpuFallback)
        } else if max_gpu_layers >= total_layers {
            (total_layers, AllocationStrategy::MaxGpu)
        } else if gpu_info.total_vram_mb >= 3500
            && max_gpu_layers >= (total_layers as f32 * 0.8) as usize
        {
            // Use aggressive strategy for high VRAM systems that can fit 80%+ of layers
            (max_gpu_layers, AllocationStrategy::Aggressive)
        } else if max_gpu_layers >= total_layers / 2 {
            (max_gpu_layers, AllocationStrategy::Balanced)
        } else {
            (max_gpu_layers, AllocationStrategy::Conservative)
        };

        let cpu_layers = total_layers.saturating_sub(gpu_layers);
        let estimated_vram_usage_mb = gpu_layers as u64 * vram_per_layer_mb + reserved_vram;

        LayerAllocation {
            gpu_layers,
            cpu_layers,
            total_layers,
            estimated_vram_usage_mb,
            allocation_strategy: strategy,
        }
    }

    /// Single entry point for planning an allocation from real model facts.
    ///
    /// Both `load_model` (`llm.rs`) and `GET /api/gpu/allocation`
    /// (`main.rs`) call this instead of `calculate_optimal_layers_v2`
    /// directly, so the layer count and per-layer VRAM estimate the
    /// allocation report shows are exactly what generation will use.
    pub fn plan_for_model(
        &self,
        gpu_info: &GpuMemoryInfo,
        facts: &ModelFacts,
        vram_limit_gb: Option<f32>,
    ) -> LayerAllocation {
        self.calculate_optimal_layers_v2(
            gpu_info,
            &facts.path,
            facts.size_mb(),
            facts.layer_count as usize,
            vram_limit_gb,
        )
    }

    /// Converts the config's `vram_limit_gb` (0 means "no limit") into the
    /// `Option<f32>` the allocation methods take.
    pub fn vram_limit_from_config(vram_limit_gb: usize) -> Option<f32> {
        if vram_limit_gb > 0 {
            Some(vram_limit_gb as f32)
        } else {
            None
        }
    }

    /// Calculate optimal GPU layer allocation (legacy method for backward compatibility)
    #[allow(dead_code)]
    pub fn calculate_optimal_layers(
        &self,
        gpu_info: &GpuMemoryInfo,
        model_size_mb: u64,
        total_layers: usize,
        vram_limit_gb: Option<f32>,
    ) -> LayerAllocation {
        // Apply VRAM limit if specified
        let effective_available_vram = if let Some(limit_gb) = vram_limit_gb {
            let limit_mb = (limit_gb * 1024.0) as u64;
            std::cmp::min(gpu_info.available_vram_mb, limit_mb)
        } else {
            gpu_info.available_vram_mb
        };

        // Apply safety margin
        let safe_vram_mb = ((effective_available_vram as f32) * self.safety_margin_percent) as u64;
        let usable_vram_mb = safe_vram_mb.saturating_sub(self.min_free_vram_mb);

        // Estimate VRAM usage per layer (rough approximation)
        let vram_per_layer_mb = if total_layers > 0 {
            model_size_mb / total_layers as u64
        } else {
            model_size_mb
        };

        // Calculate maximum layers that fit in VRAM
        let max_gpu_layers = if vram_per_layer_mb > 0 && usable_vram_mb > 0 {
            std::cmp::min((usable_vram_mb / vram_per_layer_mb) as usize, total_layers)
        } else {
            0
        };

        // Determine allocation strategy
        let (gpu_layers, strategy) = if max_gpu_layers == 0 {
            (0, AllocationStrategy::CpuFallback)
        } else if max_gpu_layers >= total_layers {
            (total_layers, AllocationStrategy::MaxGpu)
        } else if max_gpu_layers >= total_layers / 2 {
            (max_gpu_layers, AllocationStrategy::Balanced)
        } else {
            (max_gpu_layers, AllocationStrategy::Conservative)
        };

        let cpu_layers = total_layers.saturating_sub(gpu_layers);
        let estimated_vram_usage_mb = gpu_layers as u64 * vram_per_layer_mb;

        LayerAllocation {
            gpu_layers,
            cpu_layers,
            total_layers,
            estimated_vram_usage_mb,
            allocation_strategy: strategy,
        }
    }

    /// Monitor memory usage and suggest reallocation if needed
    #[allow(dead_code)]
    pub fn monitor_and_suggest_reallocation(
        &self,
        current_allocation: &LayerAllocation,
        current_gpu_info: &GpuMemoryInfo,
    ) -> Option<LayerAllocation> {
        let memory_pressure = current_gpu_info.utilization_percent;
        let available_ratio =
            current_gpu_info.available_vram_mb as f32 / current_gpu_info.total_vram_mb as f32;

        // If memory pressure is high (>90%) or available memory is low (<10%), suggest reallocation
        if memory_pressure > 90.0 || available_ratio < 0.1 {
            if current_allocation.gpu_layers > 0 {
                // Reduce GPU layers by 20%
                let new_gpu_layers = (current_allocation.gpu_layers as f32 * 0.8) as usize;
                let new_cpu_layers = current_allocation
                    .total_layers
                    .saturating_sub(new_gpu_layers);

                return Some(LayerAllocation {
                    gpu_layers: new_gpu_layers,
                    cpu_layers: new_cpu_layers,
                    total_layers: current_allocation.total_layers,
                    estimated_vram_usage_mb: new_gpu_layers as u64
                        * (current_allocation.estimated_vram_usage_mb
                            / current_allocation.gpu_layers.max(1) as u64),
                    allocation_strategy: AllocationStrategy::Conservative,
                });
            }
        }
        // If memory pressure is low (<50%) and we have available VRAM, suggest more GPU layers
        else if memory_pressure < 50.0
            && available_ratio > 0.3
            && current_allocation.cpu_layers > 0
        {
            // Increase GPU layers by 10%
            let additional_layers = (current_allocation.total_layers as f32 * 0.1) as usize;
            let new_gpu_layers = std::cmp::min(
                current_allocation.gpu_layers + additional_layers,
                current_allocation.total_layers,
            );
            let new_cpu_layers = current_allocation
                .total_layers
                .saturating_sub(new_gpu_layers);

            return Some(LayerAllocation {
                gpu_layers: new_gpu_layers,
                cpu_layers: new_cpu_layers,
                total_layers: current_allocation.total_layers,
                estimated_vram_usage_mb: new_gpu_layers as u64
                    * (current_allocation.estimated_vram_usage_mb
                        / current_allocation.gpu_layers.max(1) as u64),
                allocation_strategy: if new_gpu_layers == current_allocation.total_layers {
                    AllocationStrategy::MaxGpu
                } else {
                    AllocationStrategy::Balanced
                },
            });
        }

        None
    }

    /// Get recommended settings for different scenarios
    #[allow(dead_code)]
    pub fn get_recommended_settings(gpu_info: &GpuMemoryInfo) -> (f32, u64) {
        let safety_margin = if gpu_info.total_vram_mb > 16000 {
            0.9 // High-end GPU: can use 90%
        } else if gpu_info.total_vram_mb > 8000 {
            0.8 // Mid-range GPU: use 80%
        } else {
            0.7 // Low-end GPU: use 70%
        };

        let min_free_vram = if gpu_info.total_vram_mb > 16000 {
            1024 // Keep 1GB free for high-end
        } else if gpu_info.total_vram_mb > 8000 {
            512 // Keep 512MB free for mid-range
        } else {
            256 // Keep 256MB free for low-end
        };

        (safety_margin, min_free_vram)
    }
}

impl std::fmt::Display for LayerAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU Layers: {}/{} | Strategy: {:?} | Est. VRAM: {}MB",
            self.gpu_layers,
            self.total_layers,
            self.allocation_strategy,
            self.estimated_vram_usage_mb
        )
    }
}

impl std::fmt::Display for GpuMemoryInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {:.1}GB total, {:.1}GB available ({:.1}% utilization)",
            self.device_name,
            self.total_vram_mb as f32 / 1024.0,
            self.available_vram_mb as f32 / 1024.0,
            self.utilization_percent
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_info(total_vram_mb: u64, available_vram_mb: u64) -> GpuMemoryInfo {
        GpuMemoryInfo {
            total_vram_mb,
            available_vram_mb,
            used_vram_mb: total_vram_mb.saturating_sub(available_vram_mb),
            utilization_percent: 25.0,
            device_name: "Test GPU".to_string(),
            driver_version: "1.0".to_string(),
        }
    }

    #[test]
    fn test_calculate_optimal_layers() {
        let allocator = GpuAllocator::new();
        let gpu_info = GpuMemoryInfo {
            total_vram_mb: 8192,
            available_vram_mb: 6144,
            used_vram_mb: 2048,
            utilization_percent: 25.0,
            device_name: "Test GPU".to_string(),
            driver_version: "1.0".to_string(),
        };

        let allocation = allocator.calculate_optimal_layers(&gpu_info, 4096, 32, None);

        assert!(allocation.gpu_layers > 0);
        assert!(allocation.gpu_layers <= 32);
        assert_eq!(allocation.gpu_layers + allocation.cpu_layers, 32);
    }

    /// Regression test for the unit bug where `vram_per_layer_mb` was
    /// divided by `1024.0 * 1024.0` twice (once via `bytes_per_param`
    /// cancelling out, once explicitly), making it 1 MB for every model
    /// and forcing `MaxGpu` regardless of how large the model actually is.
    #[test]
    fn vram_per_layer_reflects_real_model_size() {
        let allocator = GpuAllocator::new();
        let info = gpu_info(8192, 8192);

        let allocation = allocator.calculate_optimal_layers_v2(&info, "model.gguf", 4096, 32, None);

        // 4096 MB / 32 layers = 128 MB/layer, comfortably fits, so this
        // should resolve to MaxGpu with all 32 layers on GPU.
        assert!(matches!(
            allocation.allocation_strategy,
            AllocationStrategy::MaxGpu
        ));
        assert_eq!(allocation.gpu_layers, 32);
        // estimated_vram_usage_mb = gpu_layers * ceil(size/layers) + 300;
        // 32 * 128 + 300 = 4396. Under the old bug this would have been
        // 32 * 1 + 300 = 332.
        assert_eq!(allocation.estimated_vram_usage_mb, 4396);
    }

    #[test]
    fn partial_offload_for_model_that_barely_fits() {
        let allocator = GpuAllocator::new();
        // ~4.9GB model, 32 layers, only 5GB free out of 6GB total.
        let info = gpu_info(6144, 5000);

        let allocation = allocator.calculate_optimal_layers_v2(&info, "model.gguf", 4915, 32, None);

        assert!(!matches!(
            allocation.allocation_strategy,
            AllocationStrategy::MaxGpu
        ));
        assert!(allocation.gpu_layers > 0 && allocation.gpu_layers < 32);
    }

    #[test]
    fn thirteen_b_model_offloads_fewer_than_all_layers_on_6gb() {
        let allocator = GpuAllocator::new();
        // ~8GB (13B Q4) model, 80 layers, 6GB free.
        let info = gpu_info(6144, 6144);

        let allocation = allocator.calculate_optimal_layers_v2(&info, "model.gguf", 8192, 80, None);

        assert!(allocation.gpu_layers < 80);
    }

    #[test]
    fn one_b_model_fully_offloads_on_6gb() {
        let allocator = GpuAllocator::new();
        // ~1GB model, 16 layers, 6GB free: should fit entirely.
        let info = gpu_info(6144, 6144);

        let allocation = allocator.calculate_optimal_layers_v2(&info, "model.gguf", 1024, 16, None);

        assert!(matches!(
            allocation.allocation_strategy,
            AllocationStrategy::MaxGpu
        ));
        assert_eq!(allocation.gpu_layers, 16);
    }

    #[test]
    fn seventy_b_model_falls_back_toward_cpu_on_6gb() {
        let allocator = GpuAllocator::new();
        // ~40GB (70B) model, 80 layers, only 6GB free.
        let info = gpu_info(6144, 6144);

        let allocation =
            allocator.calculate_optimal_layers_v2(&info, "model.gguf", 40960, 80, None);

        assert!(allocation.gpu_layers < 40); // well under half the layers
        assert!(matches!(
            allocation.allocation_strategy,
            AllocationStrategy::Conservative | AllocationStrategy::CpuFallback
        ));
    }

    #[test]
    fn plan_for_model_matches_calculate_optimal_layers_v2() {
        let allocator = GpuAllocator::new();
        let info = gpu_info(8192, 8192);
        let facts = ModelFacts {
            path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            layer_count: 32,
            file_size_bytes: 4096 * 1024 * 1024,
        };

        let via_facts = allocator.plan_for_model(&info, &facts, None);
        let direct =
            allocator.calculate_optimal_layers_v2(&info, &facts.path, facts.size_mb(), 32, None);

        assert_eq!(via_facts.gpu_layers, direct.gpu_layers);
        assert_eq!(
            via_facts.estimated_vram_usage_mb,
            direct.estimated_vram_usage_mb
        );
    }

    #[test]
    fn vram_limit_from_config_zero_means_unlimited() {
        assert_eq!(GpuAllocator::vram_limit_from_config(0), None);
        assert_eq!(GpuAllocator::vram_limit_from_config(8), Some(8.0));
    }

    #[test]
    fn test_memory_pressure_detection() {
        let allocator = GpuAllocator::new();
        let current_allocation = LayerAllocation {
            gpu_layers: 20,
            cpu_layers: 12,
            total_layers: 32,
            estimated_vram_usage_mb: 3000,
            allocation_strategy: AllocationStrategy::Balanced,
        };

        let high_pressure_info = GpuMemoryInfo {
            total_vram_mb: 8192,
            available_vram_mb: 512,
            used_vram_mb: 7680,
            utilization_percent: 95.0,
            device_name: "Test GPU".to_string(),
            driver_version: "1.0".to_string(),
        };

        let suggestion =
            allocator.monitor_and_suggest_reallocation(&current_allocation, &high_pressure_info);
        assert!(suggestion.is_some());

        if let Some(new_allocation) = suggestion {
            assert!(new_allocation.gpu_layers < current_allocation.gpu_layers);
        }
    }
}
