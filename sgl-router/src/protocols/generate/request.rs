// Generate API request types (/generate)

use crate::protocols::common::{GenerationRequest, LoRAPath, StringOrArray};
use crate::protocols::generate::types::{GenerateParameters, InputIds, SamplingParams};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// The prompt to generate from (OpenAI style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<StringOrArray>,

    /// Text input - SGLang native format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Input IDs for tokenized input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<InputIds>,

    /// Generation parameters
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<GenerateParameters>,

    /// Sampling parameters (sglang style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<SamplingParams>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Whether to return logprobs
    #[serde(default)]
    pub return_logprob: bool,

    // ============= SGLang Extensions =============
    /// Path to LoRA adapter(s) for model customization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<LoRAPath>,

    /// Session parameters for continual prompting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_params: Option<HashMap<String, serde_json::Value>>,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// Request ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<String>,
}

impl GenerationRequest for GenerateRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        // Generate requests typically don't have a model field
        None
    }

    fn extract_text_for_routing(&self) -> String {
        // Check fields in priority order: text, prompt, inputs
        if let Some(ref text) = self.text {
            return text.clone();
        }

        if let Some(ref prompt) = self.prompt {
            return match prompt {
                StringOrArray::String(s) => s.clone(),
                StringOrArray::Array(v) => v.join(" "),
            };
        }

        if let Some(ref input_ids) = self.input_ids {
            return match input_ids {
                InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| id.to_string())
                    .collect::<Vec<String>>()
                    .join(" "),
                InputIds::Batch(batches) => batches
                    .iter()
                    .flat_map(|batch| batch.iter().map(|&id| id.to_string()))
                    .collect::<Vec<String>>()
                    .join(" "),
            };
        }

        // No text input found
        String::new()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum PromptOrTokens {
    #[serde(rename = "prompts")]
    Prompts(Vec<String>),
    #[serde(rename = "prompt_tokens")]
    PromptTokens(Vec<HashMap<String, String>>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TopKOrTopP {
    #[serde(rename = "top_k")]
    TopK(u32),
    #[serde(rename = "top_p")]
    TopP(f32),
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TimeoutOrExpiry {
    #[serde(rename = "timeout_sec")]
    TimeoutSec(f32),
    #[serde(rename = "expiry_ts")]
    ExpiryTs(f32),
}

fn default_tokens_to_generate() -> u32 {
    64
}

fn default_temperature() -> f32 {
    1.0
}

fn default_stop() -> Vec<String> {
    vec![]
}

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct RawInferenceRequest {
    #[serde(flatten)]
    prompt_or_tokens: PromptOrTokens,
    #[serde(flatten)]
    top_k_or_top_p: TopKOrTopP,
    #[serde(flatten)]
    timeout_or_expiry: TimeoutOrExpiry,
    #[serde(default = "default_tokens_to_generate")]
    tokens_to_generate: u32,
    #[serde(default)]
    logprobs: bool,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_stop")]
    stop: Vec<String>,
    prompt_prefix: Option<String>,
    trim_start_str: Option<String>,
    random_seed: Option<u64>,
    fields: Option<Vec<String>>,
    stream_nth_token: Option<u32>,
    batch_ts: Option<f32>,
}

impl From<RawInferenceRequest> for GenerateRequest {
    fn from(value: RawInferenceRequest) -> Self {
        let (prompt, tokens): (Option<String>, Option<InputIds>) = match value.prompt_or_tokens {
            PromptOrTokens::Prompts(prompts) => (Some(prompts[0].clone()), None),
            PromptOrTokens::PromptTokens(_) => unreachable!(),
        };
        let (top_k, top_p): (Option<u32>, Option<f32>) = match value.top_k_or_top_p {
            TopKOrTopP::TopK(top_k) => (Some(top_k), None),
            TopKOrTopP::TopP(top_p) => (None, Some(top_p)),
        };

        Self {
            prompt: None,
            text: prompt,
            input_ids: tokens,
            parameters: Some(GenerateParameters {
                best_of: None,
                decoder_input_details: None,
                details: None,
                do_sample: None,
                max_new_tokens: Some(value.tokens_to_generate),
                seed: value.random_seed,
                repetition_penalty: None,
                stop: Some(value.stop),
                temperature: Some(value.temperature),
                top_k,
                top_p,
                return_full_text: None,
                truncate: None,
                typical_p: None,
                watermark: None,
            }),
            sampling_params: None,
            stream: false,
            return_logprob: true,
            lora_path: None,
            session_params: None,
            return_hidden_states: false,
            rid: None,
        }
    }
}
