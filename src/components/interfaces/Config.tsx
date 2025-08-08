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
}
