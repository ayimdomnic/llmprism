//! Serde structs mirroring Gemini's `generateContent`/`streamGenerateContent`
//! wire format (`POST /v1beta/models/{model}:generateContent`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    /// Both this crate's own [`Tool`](crate::Tool)s (wrapped in one
    /// `{"functionDeclarations": [...]}` entry) and any provider-native
    /// tools from
    /// [`TextRequest::provider_tools`](crate::text::TextRequest::provider_tools)
    /// (each its own entry, e.g. `{"googleSearch": {}}`) live in this one
    /// array -- Gemini tells different tool kinds apart by which key is
    /// present on each entry, the same convention [`Part`] uses for content.
    /// See [`super::maps::build_tools`].
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
pub struct SystemInstruction {
    pub parts: Vec<Part>,
}

/// One piece of a [`Content`]'s `parts` array. Gemini has no `"type"`
/// discriminator field the way Anthropic's content blocks do -- each variant
/// is told apart by which key is present, so this is `#[serde(untagged)]`
/// rather than internally tagged. [`Part::Other`] must stay last: it matches
/// any JSON object at all, so it's only tried once every named shape has
/// already failed to match.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCallPart,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponsePart,
    },
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineDataPart,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: FileDataPart,
    },
    /// Anything else this crate doesn't produce/translate (`executionResult`,
    /// thought parts, ...) -- kept so a response containing one of these
    /// doesn't fail to decode; safely ignored wherever parts are read.
    Other(Value),
}

/// Media embedded directly in the request/response as base64 -- what a
/// [`MediaPart`](crate::value_objects::MediaPart) with
/// [`MediaData::Base64`](crate::value_objects::MediaData::Base64) becomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineDataPart {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub data: String,
}

/// Media referenced by URI rather than embedded -- what a
/// [`MediaPart`](crate::value_objects::MediaPart) with
/// [`MediaData::Url`](crate::value_objects::MediaData::Url) becomes. Despite
/// the name, Gemini 2.5+ models accept an ordinary public `https://` URL
/// here directly, not just a URI from Gemini's own Files API -- earlier
/// (2.0-family) models don't support external URLs this way and need the
/// bytes embedded instead (use [`MediaData::Base64`](crate::value_objects::MediaData::Base64)).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDataPart {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    #[serde(rename = "fileUri")]
    pub file_uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallPart {
    pub name: String,
    #[serde(default)]
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponsePart {
    pub name: String,
    pub response: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// The *standard* JSON Schema field, not Gemini's older `responseSchema`
    /// (which only accepts a restricted, capitalized-type OpenAPI-3.0
    /// subset). Using this one means [`crate::schema::to_json_schema`]'s
    /// output can be sent as-is, the same way it's reused for OpenAI and
    /// Anthropic, instead of needing a whole second schema dialect just for
    /// Gemini.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_json_schema: Option<Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ToolDeclaration {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Serialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    /// Same reasoning as [`GenerationConfig::response_json_schema`]: the
    /// standard-JSON-Schema field, not the older capitalized-type
    /// `parameters` field.
    #[serde(rename = "parametersJsonSchema")]
    pub parameters_json_schema: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    pub mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub usage_metadata: Option<UsageMetadata>,
    #[serde(default)]
    pub model_version: Option<String>,
    #[serde(default)]
    pub response_id: Option<String>,
    #[serde(default)]
    pub prompt_feedback: Option<PromptFeedback>,
}

#[derive(Debug, Deserialize)]
pub struct Candidate {
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(default, rename = "finishReason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(default)]
    pub prompt_token_count: u32,
    #[serde(default)]
    pub candidates_token_count: u32,
    #[serde(default)]
    pub cached_content_token_count: Option<u32>,
}

/// Present instead of `candidates` when Gemini declines to generate anything
/// at all for the request (most commonly a safety block) -- see
/// [`super::maps::parse_response`].
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    #[serde(default)]
    pub block_reason: Option<String>,
}
