//! Image generation. Start with [`crate::Registry::images`].

pub mod request;
pub mod response;

pub use request::{ImagesRequest, PendingImagesRequest};
pub use response::{GeneratedImage, ImagesResponse};
