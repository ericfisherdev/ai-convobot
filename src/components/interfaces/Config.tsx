export enum Device {
    CPU = "CPU",
    GPU = "GPU",
    Metal = "Metal"
}

export enum PromptTemplate {
    Auto = "Auto",
    Default = "Default",
    Llama2 = "Llama2",
    Mistral = "Mistral"
}

export enum MultiplayerMode {
    Solo = "solo",
    Host = "host",
    Joiner = "joiner"
}

export interface ConfigInterface {
    device: Device;
    llm_model_path: string;
    selected_model_path?: string;
    gpu_layers: number;
    prompt_template: PromptTemplate;
    context_window_size: number;
    max_response_tokens: number;
    enable_dynamic_context: boolean;
    vram_limit_gb: number;
    dynamic_gpu_allocation: boolean;
    gpu_safety_margin: number;
    min_free_vram_mb: number;
    multiplayer_mode: MultiplayerMode;
    multiplayer_password_set: boolean;
    multiplayer_host_address: string;
    multiplayer_participant_id: string;
    mention_followup_depth: number;
    remote_generation_timeout_secs: number;
    /** Write-only: only ever populated by the form and sent on PUT. */
    multiplayer_password?: string;
    /** `null` means "derive at runtime from the recent-message budget". */
    compact_threshold_tokens: number | null;
    compact_min_messages: number;
    /** `null` means "use llm_model_path". */
    compaction_model_path: string | null;
    heuristic_person_detection: boolean;
    /** How far a compaction commit blends the running attitude toward the
     * narrative rating: 0 keeps the running values, 1 adopts the rating. */
    compaction_attitude_weight: number;
    /** After each round, the companion writes a short private note about
     * what it took from it, and replies in light of that note. */
    running_thoughts_enabled: boolean;
}

export interface ModelInfo {
    path: string;
    filename: string;
    size_bytes: number;
    directory: string;
    last_modified: string;
}

export interface DirectoryInfo {
    id: number;
    path: string;
    created_at: string;
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
    allocation_strategy: "MaxGpu" | "Balanced" | "Conservative" | "CpuFallback" | "Aggressive";
    model?: {
        path: string;
        architecture: string;
        layer_count: number;
        file_size_bytes: number;
    };
}
