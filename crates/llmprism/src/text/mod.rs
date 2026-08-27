//! Text generation -- the "chat with a model, optionally letting it call tools"
//! capability. Start with [`crate::Registry::text`], which hands you a
//! [`PendingTextRequest`] to configure and run.

pub mod request;
pub mod response;

pub use request::{PendingTextRequest, TextRequest, ToolChoice};
pub use response::{Step, TextResponse};
