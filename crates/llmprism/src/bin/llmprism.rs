//! `llmprism` -- a command-line wrapper around the `llmprism` library, so
//! every capability it supports (text generation, streaming, structured
//! output, moderation, embeddings, rerank, images, audio) is reachable
//! straight from the shell, not just from Rust code.
//!
//! A thin wrapper, deliberately: [`Registry::from_env`] already only
//! registers a provider when both its Cargo feature and its API-key
//! environment variable are present, so this binary needs no
//! provider-specific logic of its own -- it just calls the same public API
//! any other `llmprism` consumer would.
//!
//! Build/install with the providers you want compiled in, e.g.:
//!
//! ```sh
//! cargo install llmprism --features cli,openai,anthropic
//! ```

use std::io::Read;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use futures::StreamExt;
use llmprism::schema::ObjectSchema;
use llmprism::text::PendingTextRequest;
use llmprism::{Registry, StreamEvent};
use serde::Serialize;

#[derive(Parser)]
#[command(
    name = "llmprism",
    version,
    about = "Talk to LLM providers from the command line."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate text.
    Text(TextArgs),
    /// Generate text, printing the reply as it streams in.
    Stream(TextArgs),
    /// Generate a reply matching a JSON Schema.
    Structured(StructuredArgs),
    /// Check text against a provider's content-safety classifier.
    Moderate(ModerateArgs),
    /// Turn text into embedding vectors.
    Embed(EmbedArgs),
    /// Score and sort documents by relevance to a query.
    Rerank(RerankArgs),
    /// Generate an image from a text prompt.
    Image(ImageArgs),
    /// Turn text into spoken audio.
    Speak(SpeakArgs),
    /// Transcribe spoken audio into text.
    Transcribe(TranscribeArgs),
    /// List providers compiled in and configured with an API key.
    Providers,
}

#[derive(Args)]
struct CommonArgs {
    /// Provider to use (e.g. "openai", "anthropic").
    #[arg(short, long)]
    provider: String,
    /// Model to use (e.g. "gpt-4o-mini").
    #[arg(short, long)]
    model: String,
    /// Print the full response as JSON instead of a plain-text summary.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct TextArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// The prompt to send. Reads from stdin if omitted.
    #[arg(short = 'P', long)]
    prompt: Option<String>,
    /// A system-level instruction.
    #[arg(short, long)]
    system: Option<String>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    max_tokens: Option<u32>,
    /// How many tool-calling round trips to allow. Only meaningful with
    /// --mcp-stdio/--mcp-http attached; irrelevant otherwise.
    #[arg(long, default_value_t = 1)]
    max_steps: u32,
    /// Connect to an MCP server over stdio and attach its tools -- the
    /// command and its arguments as one string (e.g. "npx -y some-server").
    /// Repeatable.
    #[cfg(feature = "mcp")]
    #[arg(long = "mcp-stdio")]
    mcp_stdio: Vec<String>,
    /// Connect to an MCP server over Streamable HTTP and attach its tools.
    /// Repeatable.
    #[cfg(feature = "mcp")]
    #[arg(long = "mcp-http")]
    mcp_http: Vec<String>,
}

#[derive(Args)]
struct StructuredArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(short = 'P', long)]
    prompt: Option<String>,
    #[arg(short, long)]
    system: Option<String>,
    /// Path to a JSON Schema file describing the required reply shape.
    #[arg(long)]
    schema_file: std::path::PathBuf,
}

#[derive(Args)]
struct ModerateArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Text to check. Reads from stdin if omitted.
    #[arg(short = 'i', long)]
    input: Option<String>,
}

#[derive(Args)]
struct EmbedArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Text to embed. Pass more than once for several inputs in one call.
    /// Reads a single input from stdin if none are given.
    #[arg(short = 'i', long = "input")]
    inputs: Vec<String>,
    #[arg(long)]
    dimensions: Option<u32>,
}

#[derive(Args)]
struct RerankArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(short, long)]
    query: String,
    /// A document to score. Pass more than once.
    #[arg(short, long = "document")]
    documents: Vec<String>,
    #[arg(long)]
    top_k: Option<u32>,
}

#[derive(Args)]
struct ImageArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(short = 'P', long)]
    prompt: Option<String>,
    #[arg(long)]
    size: Option<String>,
    #[arg(long)]
    quality: Option<String>,
    #[arg(long)]
    style: Option<String>,
}

#[derive(Args)]
struct SpeakArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Text to speak. Reads from stdin if omitted.
    #[arg(short = 'i', long)]
    input: Option<String>,
    #[arg(long)]
    voice: Option<String>,
    /// Where to write the generated audio.
    #[arg(short, long)]
    out: std::path::PathBuf,
}

#[derive(Args)]
struct TranscribeArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Path to the audio file to transcribe.
    #[arg(short, long)]
    file: std::path::PathBuf,
}

// `current_thread`, not the default `multi_thread`: this binary runs one
// command and exits, with no need for a thread pool, and it keeps the
// crate's own `tokio` dependency (used by the library too, where a
// long-running server *would* want `multi_thread`) from having to pull in
// `rt-multi-thread` just to satisfy this binary.
#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let registry = Registry::from_env();

    match cli.command {
        Command::Text(args) => run_text(&registry, args, false).await,
        Command::Stream(args) => run_text(&registry, args, true).await,
        Command::Structured(args) => run_structured(&registry, args).await,
        Command::Moderate(args) => run_moderate(&registry, args).await,
        Command::Embed(args) => run_embed(&registry, args).await,
        Command::Rerank(args) => run_rerank(&registry, args).await,
        Command::Image(args) => run_image(&registry, args).await,
        Command::Speak(args) => run_speak(&registry, args).await,
        Command::Transcribe(args) => run_transcribe(&registry, args).await,
        Command::Providers => run_providers(&registry),
    }
}

/// Reads `value` if given, otherwise the whole of stdin (trimmed of a
/// trailing newline) -- the fallback that makes every text-taking command
/// pipeable, e.g. `echo "..." | llmprism text -p openai -m gpt-4o-mini`.
fn value_or_stdin(value: Option<String>) -> Result<String, std::io::Error> {
    match value {
        Some(value) => Ok(value),
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            Ok(buf.trim_end_matches('\n').to_string())
        }
    }
}

fn print_result<T: Serialize>(value: &T, summary: impl FnOnce(&T) -> String, json: bool) {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(value).expect("response types always serialize")
        );
    } else {
        println!("{}", summary(value));
    }
}

/// The MCP connections a text/stream request attached, if any -- kept alive
/// by the caller until the request finishes, so each connection shuts down
/// normally afterward instead of being force-killed by process exit.
#[cfg(feature = "mcp")]
type McpConnections = Vec<llmprism::mcp::McpToolset>;
#[cfg(not(feature = "mcp"))]
type McpConnections = ();

async fn build_text_request(
    registry: &Registry,
    args: &TextArgs,
) -> Result<(PendingTextRequest, McpConnections), Box<dyn std::error::Error>> {
    let prompt = value_or_stdin(args.prompt.clone())?;
    let mut request = registry.text(&args.common.provider, &args.common.model)?;

    if let Some(system) = &args.system {
        request = request.with_system_prompt(system);
    }
    request = request.with_prompt(prompt);
    if let Some(temperature) = args.temperature {
        request = request.with_temperature(temperature);
    }
    if let Some(max_tokens) = args.max_tokens {
        request = request.with_max_tokens(max_tokens);
    }
    request = request.with_max_steps(args.max_steps);

    #[cfg(feature = "mcp")]
    let mut connections = Vec::new();

    #[cfg(feature = "mcp")]
    {
        for target in &args.mcp_stdio {
            let mut parts = target.split_whitespace();
            let command = parts
                .next()
                .ok_or("--mcp-stdio needs a command, e.g. \"npx -y some-server\"")?;
            let toolset = llmprism::mcp::McpToolset::connect_stdio(command, parts).await?;
            request = request.with_tools(toolset.tools());
            connections.push(toolset);
        }
        for url in &args.mcp_http {
            let toolset = llmprism::mcp::McpToolset::connect_http(url.clone()).await?;
            request = request.with_tools(toolset.tools());
            connections.push(toolset);
        }
    }

    #[cfg(not(feature = "mcp"))]
    let connections = ();

    Ok((request, connections))
}

async fn run_text(
    registry: &Registry,
    args: TextArgs,
    stream: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = args.common.json;
    let (request, _connections) = build_text_request(registry, &args).await?;

    if stream {
        let mut events = request.stream();
        let mut final_response = None;
        while let Some(event) = events.next().await {
            match event? {
                StreamEvent::TextDelta { text } => {
                    if !json {
                        print!("{text}");
                        use std::io::Write;
                        std::io::stdout().flush().ok();
                    }
                }
                StreamEvent::StreamEnd { response } => final_response = Some(response),
                _ => {}
            }
        }
        if !json {
            println!();
        }
        if let Some(response) = final_response {
            if json {
                print_result(&response, |_| String::new(), true);
            }
        }
    } else {
        let response = request.generate().await?;
        print_result(&response, |r| r.text.clone().unwrap_or_default(), json);
    }

    Ok(())
}

async fn run_structured(
    registry: &Registry,
    args: StructuredArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = value_or_stdin(args.prompt)?;
    let schema_text = std::fs::read_to_string(&args.schema_file)
        .map_err(|e| format!("{}: {e}", args.schema_file.display()))?;
    let schema_json: serde_json::Value = serde_json::from_str(&schema_text)
        .map_err(|e| format!("{}: {e}", args.schema_file.display()))?;
    let schema = ObjectSchema::from_raw_json_schema("response", schema_json);

    let mut request = registry.structured(&args.common.provider, &args.common.model, schema)?;
    if let Some(system) = &args.system {
        request = request.with_system_prompt(system);
    }
    let response = request.with_prompt(prompt).generate().await?;

    print_result(
        &response,
        |r| serde_json::to_string_pretty(&r.data).unwrap_or_default(),
        args.common.json,
    );
    Ok(())
}

async fn run_moderate(
    registry: &Registry,
    args: ModerateArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = value_or_stdin(args.input)?;
    let response = registry
        .moderation(&args.common.provider, &args.common.model)?
        .with_input(input)
        .generate()
        .await?;

    print_result(
        &response,
        |r| {
            r.results
                .iter()
                .map(|result| format!("flagged: {}", result.flagged))
                .collect::<Vec<_>>()
                .join("\n")
        },
        args.common.json,
    );
    Ok(())
}

async fn run_embed(registry: &Registry, args: EmbedArgs) -> Result<(), Box<dyn std::error::Error>> {
    let inputs = if args.inputs.is_empty() {
        vec![value_or_stdin(None)?]
    } else {
        args.inputs
    };

    let mut request = registry.embeddings(&args.common.provider, &args.common.model)?;
    for input in inputs {
        request = request.with_input(input);
    }
    if let Some(dimensions) = args.dimensions {
        request = request.with_dimensions(dimensions);
    }
    let response = request.generate().await?;

    print_result(
        &response,
        |r| {
            format!(
                "{} embedding(s), {} dimension(s) each",
                r.embeddings.len(),
                r.embeddings.first().map(|e| e.vector.len()).unwrap_or(0)
            )
        },
        args.common.json,
    );
    Ok(())
}

async fn run_rerank(
    registry: &Registry,
    args: RerankArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut request = registry.rerank(&args.common.provider, &args.common.model, args.query)?;
    request = request.with_documents(args.documents);
    if let Some(top_k) = args.top_k {
        request = request.with_top_k(top_k);
    }
    let response = request.generate().await?;

    print_result(
        &response,
        |r| {
            r.results
                .iter()
                .map(|result| format!("[{}] {:.4}", result.index, result.relevance_score))
                .collect::<Vec<_>>()
                .join("\n")
        },
        args.common.json,
    );
    Ok(())
}

async fn run_image(registry: &Registry, args: ImageArgs) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = value_or_stdin(args.prompt)?;
    let mut request = registry.images(&args.common.provider, &args.common.model, prompt)?;
    if let Some(size) = args.size {
        request = request.with_size(size);
    }
    if let Some(quality) = args.quality {
        request = request.with_quality(quality);
    }
    if let Some(style) = args.style {
        request = request.with_style(style);
    }
    let response = request.generate().await?;

    print_result(
        &response,
        |r| {
            r.images
                .iter()
                .map(|image| match &image.data {
                    llmprism::value_objects::MediaData::Url(url) => url.clone(),
                    llmprism::value_objects::MediaData::Base64(_) => {
                        "<base64-encoded image data -- use --json to get it>".to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        },
        args.common.json,
    );
    Ok(())
}

async fn run_speak(registry: &Registry, args: SpeakArgs) -> Result<(), Box<dyn std::error::Error>> {
    let input = value_or_stdin(args.input)?;
    let mut request = registry.text_to_speech(&args.common.provider, &args.common.model, input)?;
    if let Some(voice) = args.voice {
        request = request.with_voice(voice);
    }
    let response = request.generate().await?;

    std::fs::write(&args.out, &response.audio.data)
        .map_err(|e| format!("{}: {e}", args.out.display()))?;
    if args.common.json {
        // Deliberately not the full `AudioResponse` here -- that would dump
        // the whole audio payload as a JSON array of numbers, which is both
        // enormous and pointless once it's already on disk at `args.out`.
        let summary = serde_json::json!({
            "path": args.out,
            "bytes_written": response.audio.data.len(),
            "mime_type": response.audio.mime_type,
            "meta": response.meta,
        });
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "wrote {} bytes ({}) to {}",
            response.audio.data.len(),
            response.audio.mime_type,
            args.out.display()
        );
    }
    Ok(())
}

async fn run_transcribe(
    registry: &Registry,
    args: TranscribeArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read(&args.file).map_err(|e| format!("{}: {e}", args.file.display()))?;
    let mime_type = mime_type_for(&args.file);
    let filename = args
        .file
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "audio".to_string());

    let audio = llmprism::audio::AudioInput::new(data, mime_type).with_filename(filename);
    let response = registry
        .speech_to_text(&args.common.provider, &args.common.model, audio)?
        .generate()
        .await?;

    print_result(&response, |r| r.text.clone(), args.common.json);
    Ok(())
}

fn run_providers(registry: &Registry) -> Result<(), Box<dyn std::error::Error>> {
    let mut names = registry.provider_names();
    names.sort_unstable();
    if names.is_empty() {
        println!("no providers configured -- set an API key (e.g. OPENAI_API_KEY) and rebuild with that provider's feature enabled");
    } else {
        for name in names {
            println!("{name}");
        }
    }
    Ok(())
}

/// A best-effort MIME type from a file extension, for
/// [`llmprism::audio::AudioInput`]. Falls back to a generic audio type
/// rather than failing outright -- most providers only use this as a hint.
fn mime_type_for(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        Some("m4a") => "audio/mp4",
        Some("flac") => "audio/flac",
        Some("ogg") => "audio/ogg",
        Some("webm") => "audio/webm",
        _ => "application/octet-stream",
    }
}
