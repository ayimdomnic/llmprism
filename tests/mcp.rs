//! Live smoke test against a real MCP server -- the official "everything"
//! reference server, run locally via `npx`. Needs no API key (unlike every
//! other live test under `tests/`), but does need `npx` (from Node.js) on
//! `PATH` and network access the first time (to fetch the npm package) --
//! skips cleanly if `npx` isn't available, the same way a live provider test
//! skips cleanly without its API key.

#![cfg(feature = "mcp")]

use llmprism::mcp::McpToolset;

#[tokio::test]
async fn connects_over_stdio_and_lists_the_reference_servers_tools() {
    if std::process::Command::new("npx")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping: npx (from Node.js) isn't on PATH");
        return;
    }

    let toolset =
        McpToolset::connect_stdio("npx", ["-y", "@modelcontextprotocol/server-everything"])
            .await
            .expect("should connect to the reference MCP server over stdio");

    let tools = toolset.tools();
    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();

    assert!(
        names.contains(&"get-sum"),
        "expected the reference server's 'get-sum' tool, got: {names:?}"
    );

    let get_sum = tools
        .iter()
        .find(|t| t.name() == "get-sum")
        .expect("just asserted 'get-sum' is present");

    // The tool's parameters came from the server's own (real) JSON Schema,
    // via `ObjectSchema::from_raw_json_schema` -- confirms that mapping
    // actually round-trips into a usable schema, not just that connecting
    // and listing worked.
    let schema = llmprism::schema::to_json_schema(&llmprism::schema::Schema::Object(
        get_sum.parameters().clone(),
    ));
    assert_eq!(schema["type"], "object");
}
