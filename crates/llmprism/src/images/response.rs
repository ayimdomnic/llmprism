use serde::{Deserialize, Serialize};

use crate::value_objects::{MediaData, Meta};

/// The result of an image-generation call: one [`GeneratedImage`] per
/// requested image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagesResponse {
    pub images: Vec<GeneratedImage>,
    pub meta: Meta,
}

/// One generated image, either as a URL to fetch it from or as embedded
/// base64 data -- reuses [`MediaData`] (the same Url-or-Base64 shape a
/// message's image attachments use) rather than a near-identical type of its
/// own.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    pub data: MediaData,
    /// Some models (e.g. DALL-E 3) rewrite the prompt before generating and
    /// report what they actually used; `None` if the provider doesn't do
    /// this or didn't report it.
    pub revised_prompt: Option<String>,
}
