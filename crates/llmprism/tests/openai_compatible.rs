//! Confirms the OpenAI-compatible provider family reports its own name (not
//! `"openai"`, which it internally reuses) and can be registered/resolved
//! the normal way. No network access -- constructing a provider and reading
//! `Provider::name()` doesn't make any HTTP calls.

#![cfg(any(
    feature = "groq",
    feature = "deepseek",
    feature = "mistral",
    feature = "xai",
    feature = "openrouter",
    feature = "perplexity",
    feature = "zai",
))]

use llmprism::Provider;

#[test]
#[cfg(feature = "groq")]
fn groq_reports_its_own_name_not_openai() {
    use llmprism::providers::openai_compatible::GroqProvider;
    assert_eq!(GroqProvider::new("test-key").name(), "groq");
}

#[test]
#[cfg(feature = "deepseek")]
fn deepseek_reports_its_own_name_not_openai() {
    use llmprism::providers::openai_compatible::DeepSeekProvider;
    assert_eq!(DeepSeekProvider::new("test-key").name(), "deepseek");
}

#[test]
#[cfg(feature = "mistral")]
fn mistral_reports_its_own_name_not_openai() {
    use llmprism::providers::openai_compatible::MistralProvider;
    assert_eq!(MistralProvider::new("test-key").name(), "mistral");
}

#[test]
#[cfg(feature = "xai")]
fn xai_reports_its_own_name_not_openai() {
    use llmprism::providers::openai_compatible::XaiProvider;
    assert_eq!(XaiProvider::new("test-key").name(), "xai");
}

#[test]
#[cfg(feature = "openrouter")]
fn openrouter_reports_its_own_name_not_openai() {
    use llmprism::providers::openai_compatible::OpenRouterProvider;
    assert_eq!(OpenRouterProvider::new("test-key").name(), "openrouter");
}

#[test]
#[cfg(feature = "perplexity")]
fn perplexity_reports_its_own_name_not_openai() {
    use llmprism::providers::openai_compatible::PerplexityProvider;
    assert_eq!(PerplexityProvider::new("test-key").name(), "perplexity");
}

#[test]
#[cfg(feature = "zai")]
fn zai_reports_its_own_name_not_openai() {
    use llmprism::providers::openai_compatible::ZaiProvider;
    assert_eq!(ZaiProvider::new("test-key").name(), "zai");
}

#[test]
#[cfg(feature = "groq")]
fn with_base_url_overrides_the_default() {
    use llmprism::providers::openai_compatible::GroqProvider;

    let mut registry = llmprism::Registry::new();
    registry.register(
        "groq",
        GroqProvider::with_base_url("test-key", "http://localhost:9999/v1"),
    );

    // Just confirms registration/resolution works the normal way -- actually
    // exercising the overridden URL would require a network call, which
    // belongs in a live test, not this offline one.
    assert!(registry.provider("groq").is_ok());
}
