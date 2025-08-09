export enum Device {
    CPU = "CPU",
    GPU = "GPU",
    Metal = "Metal"
}

export enum PromptTemplate {
    Default = "Default",
    Llama2 = "Llama2",
    Mistral = "Mistral"
}

export interface ConfigInterface {
    device: Device;
    llm_model_path: string;
    gpu_layers: number;
    prompt_template: PromptTemplate;
    context_window_size: number;
    max_response_tokens: number;
    enable_dynamic_context: boolean;
    vram_limit_gb: number;
    dynamic_gpu_allocation: boolean;
    gpu_safety_margin: number;
    min_free_vram_mb: number;
}

export interface GpuMemoryInfo {
    total_vram_mb: number;
    available_vram_mb: number;
    used_vram_mb: number;
    utilization_percent: number;
    device_name: string;
    driver_version: string;
}

export interface LayerAllocation {
    gpu_layers: number;
    cpu_layers: number;
    total_layers: number;
    estimated_vram_usage_mb: number;
    allocation_strategy: "MaxGpu" | "Balanced" | "Conservative" | "CpuFallback";
}
