//! MCP (Model Context Protocol) client support -- connect to a remote MCP
//! server, discover its tools, and use them exactly like any other
//! `llmprism` [`Tool`], with no protocol-specific code in your application.
//! Built on the official [`rmcp`] SDK.
//!
//! ```no_run
//! # #[cfg(all(feature = "mcp", feature = "openai"))]
//! # async fn example() -> Result<(), llmprism::Error> {
//! use llmprism::mcp::McpToolset;
//! use llmprism::Registry;
//!
//! // Spawns a local MCP server over stdio and lists its tools.
//! let toolset = McpToolset::connect_stdio("npx", ["-y", "some-mcp-server"]).await?;
//!
//! let registry = Registry::from_env();
//! let response = registry
//!     .text("openai", "gpt-4o-mini")?
//!     .with_prompt("What files are in the current directory?")
//!     .with_tools(toolset.tools())
//!     .with_max_steps(4)
//!     .generate()
//!     .await?;
//!
//! println!("{}", response.text.unwrap_or_default());
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use rmcp::model::{CallToolRequestParams, ContentBlock};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::ServiceExt;
use serde_json::Value;

use crate::error::{Error, ToolError};
use crate::schema::ObjectSchema;
use crate::tool::Tool;
use crate::value_objects::ToolOutput;

/// A connected MCP server. Discover its tools once with [`tools`](Self::tools)
/// and attach them to a request the normal way, via
/// [`with_tool`](crate::text::PendingTextRequest::with_tool)/
/// [`with_tools`](crate::text::PendingTextRequest::with_tools) -- the model
/// (and the rest of this crate) sees an ordinary [`Tool`], with no idea it's
/// actually backed by a remote server.
///
/// Keeping this alive keeps the underlying connection (a child process, or
/// an HTTP session) open; every tool handed out by [`tools`](Self::tools)
/// holds a cheap, shared handle back to it, so the toolset itself can be
/// dropped once you're done handing out tools -- the connection stays open
/// as long as at least one of those tools is still alive.
pub struct McpToolset {
    // Never read directly -- kept only so dropping `McpToolset` before its
    // tools are (e.g. a server with no tools at all) still closes the
    // connection via this field's own `Drop`, rather than leaving nothing
    // holding it open at all.
    #[allow(dead_code)]
    service: Arc<RunningService<RoleClient, ()>>,
    tools: Vec<Arc<dyn Tool>>,
}

impl McpToolset {
    /// Spawns `command` (with `args`) as a child process and speaks MCP with
    /// it over stdio -- the way most local MCP servers (an `npx`/`uvx`
    /// package, a compiled binary) are meant to be run.
    pub async fn connect_stdio(
        command: impl AsRef<str>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, Error> {
        let mut process = tokio::process::Command::new(command.as_ref());
        process.args(args.into_iter().map(Into::into));

        let transport = TokioChildProcess::new(process).map_err(|e| Error::Mcp {
            message: format!("failed to spawn '{}': {e}", command.as_ref()),
        })?;

        let service = ().serve(transport).await.map_err(|e| Error::Mcp {
            message: e.to_string(),
        })?;

        Self::from_running_service(service).await
    }

    /// Connects to a remote MCP server over Streamable HTTP at `url`.
    pub async fn connect_http(url: impl Into<String>) -> Result<Self, Error> {
        let transport = StreamableHttpClientTransport::from_uri(url.into());

        let service = ().serve(transport).await.map_err(|e| Error::Mcp {
            message: e.to_string(),
        })?;

        Self::from_running_service(service).await
    }

    async fn from_running_service(service: RunningService<RoleClient, ()>) -> Result<Self, Error> {
        let service = Arc::new(service);

        let mcp_tools = service.list_all_tools().await.map_err(|e| Error::Mcp {
            message: e.to_string(),
        })?;

        let tools = mcp_tools
            .into_iter()
            .map(|tool| {
                let name = tool.name.to_string();
                let description = tool
                    .description
                    .map(|d| d.to_string())
                    .unwrap_or_default();
                let parameters = ObjectSchema::from_raw_json_schema(
                    name.clone(),
                    Value::Object((*tool.input_schema).clone()),
                );

                Arc::new(McpTool {
                    service: service.clone(),
                    name,
                    description,
                    parameters,
                }) as Arc<dyn Tool>
            })
            .collect();

        Ok(Self { service, tools })
    }

    /// Every tool this server offers, ready to attach to a request.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }
}

/// One tool discovered on a connected [`McpToolset`], adapted to this
/// crate's [`Tool`] trait. You won't normally construct or name this type
/// directly -- get instances from [`McpToolset::tools`].
struct McpTool {
    service: Arc<RunningService<RoleClient, ()>>,
    name: String,
    description: String,
    parameters: ObjectSchema,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> &ObjectSchema {
        &self.parameters
    }

    async fn call(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let mut params = CallToolRequestParams::new(self.name.clone());
        if let Value::Object(arguments) = args {
            params = params.with_arguments(arguments);
        }
        // A non-object `args` shouldn't happen in practice -- the tool loop
        // always sends whatever JSON the model produced for an
        // object-schema tool's arguments, which is always an object -- but
        // rather than force an explicit-empty-object arguments field onto
        // the wire for that unexpected case, just send none at all.

        let result = self
            .service
            .call_tool(params)
            .await
            .map_err(|e| ToolError::Runtime {
                name: self.name.clone(),
                message: e.to_string(),
            })?;

        let text = result
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error == Some(true) {
            return Err(ToolError::Runtime {
                name: self.name.clone(),
                message: if text.is_empty() {
                    "tool call failed with no further detail".to_string()
                } else {
                    text
                },
            });
        }

        let mut artifacts = Vec::new();
        artifacts.extend(result.structured_content);
        for block in &result.content {
            if !matches!(block, ContentBlock::Text(_)) {
                if let Ok(value) = serde_json::to_value(block) {
                    artifacts.push(value);
                }
            }
        }

        Ok(ToolOutput {
            content: text,
            artifacts,
        })
    }
}
