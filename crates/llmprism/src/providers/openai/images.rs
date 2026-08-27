//! Wire format and mapping for OpenAI's Images API
//! (`POST /v1/images/generations`) -- its own endpoint and wire shape, same
//! as moderation and embeddings.
//!
//! Deliberately doesn't set a `response_format` field: newer OpenAI image
//! models (e.g. `gpt-image-1`) don't accept it and always return base64
//! data, while older ones (the `dall-e-*` family) default to returning a
//! URL. [`parse_response`] handles whichever shape actually comes back
//! rather than requesting one and assuming the provider complied.

use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::images::{GeneratedImage, ImagesRequest, ImagesResponse};
use crate::value_objects::{MediaData, Meta};

#[derive(Debug, Serialize)]
pub struct ApiRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ApiResponse {
    pub data: Vec<ApiImage>,
}

#[derive(Debug, Deserialize)]
pub struct ApiImage {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub b64_json: Option<String>,
    #[serde(default)]
    pub revised_prompt: Option<String>,
}

pub fn build_request(request: &ImagesRequest) -> ApiRequest {
    ApiRequest {
        model: request.model.clone(),
        prompt: request.prompt.clone(),
        n: request.n,
        size: request.size.clone(),
        quality: request.quality.clone(),
        style: request.style.clone(),
    }
}

pub fn parse_response(response: ApiResponse, provider_name: &str) -> Result<ImagesResponse, Error> {
    let images = response
        .data
        .into_iter()
        .map(|image| {
            let data = match (image.url, image.b64_json) {
                (Some(url), _) => MediaData::Url(url),
                (None, Some(b64)) => MediaData::Base64(b64),
                (None, None) => {
                    return Err(Error::Decode {
                        provider: provider_name.to_string(),
                        message: "image result had neither a url nor base64 data".to_string(),
                    })
                }
            };
            Ok(GeneratedImage {
                data,
                revised_prompt: image.revised_prompt,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(ImagesResponse {
        images,
        meta: Meta::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_request_passes_quality_and_style_through() {
        let mut request = ImagesRequest::new("dall-e-3", "a lighthouse");
        request.quality = Some("hd".to_string());
        request.style = Some("vivid".to_string());

        let wire_request = build_request(&request);

        assert_eq!(wire_request.quality.as_deref(), Some("hd"));
        assert_eq!(wire_request.style.as_deref(), Some("vivid"));
    }
}
