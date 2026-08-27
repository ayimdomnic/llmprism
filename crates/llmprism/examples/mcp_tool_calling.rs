//! Connecting to a real MCP server, letting it hand its tools straight to a
//! model -- no protocol-specific code in the request itself, since
//! `McpToolset::tools()` returns ordinary [`llmprism::Tool`]s.
//!
//! Uses the official MCP "everything" reference server (a test/demo server,
//! not meant for production use, but perfect here since it needs no
//! account/API key of its own and exposes a few simple tools including
//! `get-sum` and `echo`).
//!
//! Run with:
//!
//! ```sh
//! OPENAI_API_KEY=sk-... cargo run --example mcp_tool_calling --features mcp,openai
//! ```
//!
//! Requires `npx` on your `PATH` (from Node.js) to spawn the reference
//! server -- skips cleanly if it isn't available, the same way every other
//! live example here skips cleanly without its required API key.

#[tokio::main]
async fn main() {
    #[cfg(all(feature = "mcp", feature = "openai"))]
    {
        let Ok(_) = std::env::var("OPENAI_API_KEY") else {
            eprintln!("skipping: set OPENAI_API_KEY to run this example against the real API");
            return;
        };

        if std::process::Command::new("npx")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("skipping: npx (from Node.js) isn't on PATH, needed to run the MCP server");
            return;
        }

        use llmprism::mcp::McpToolset;

        let toolset =
            McpToolset::connect_stdio("npx", ["-y", "@modelcontextprotocol/server-everything"])
                .await
                .expect("should connect to the reference MCP server over stdio");

        let registry = llmprism::Registry::from_env();

        let response = registry
            .text("openai", "gpt-4o-mini")
            .expect("openai should be registered since OPENAI_API_KEY is set")
            .with_prompt("Use the sum tool to add 12 and 30, then tell me the result.")
            .with_tools(toolset.tools())
            .with_max_steps(4)
            .generate()
            .await
            .expect("request should succeed");

        println!("{}", response.text.unwrap_or_default());
        println!("(took {} round trip(s))", response.steps.len());
    }

    #[cfg(not(all(feature = "mcp", feature = "openai")))]
    eprintln!("skipping: run with --features mcp,openai");
}
