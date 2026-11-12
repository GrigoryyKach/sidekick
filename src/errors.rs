//! Backend errors as codes: the backend only says what went wrong, `strings.rs` says it in the
//! interface language. The raw technical detail (OS error text, stderr, API message) is appended as-is.

use crate::i18n::Lang;
use crate::strings::TABLE;

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    ClaudeNotFound,
    /// Could not start the named program.
    RunFailed(String),
    NoAnthropicKey,
    NoModel,
    NoCommand,
    AgentStopped,
    AgentNotRunning,
    ModelDeclined,
    /// The custom command exited with this code; the detail is its stderr.
    CommandExit(Option<i32>),
    SttScriptMissing,
    NoMonitors,
    CaptureFailed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppError {
    pub kind: Kind,
    pub detail: String,
}

impl AppError {
    pub fn new(kind: Kind) -> anyhow::Error {
        anyhow::Error::new(AppError { kind, detail: String::new() })
    }

    pub fn with(kind: Kind, detail: impl ToString) -> anyhow::Error {
        anyhow::Error::new(AppError { kind, detail: detail.to_string() })
    }

    /// Message in the given language, detail appended.
    pub fn text(&self, lang: Lang) -> String {
        let s = TABLE.get(lang);
        let head = match &self.kind {
            Kind::ClaudeNotFound => s.err_claude_not_found.to_string(),
            Kind::RunFailed(what) => s.err_run_failed.replace("{name}", what),
            Kind::NoAnthropicKey => s.err_no_anthropic_key.to_string(),
            Kind::NoModel => s.err_no_model.to_string(),
            Kind::NoCommand => s.err_no_command.to_string(),
            Kind::AgentStopped => s.err_agent_stopped.to_string(),
            Kind::AgentNotRunning => s.err_agent_not_running.to_string(),
            Kind::ModelDeclined => s.err_model_declined.to_string(),
            Kind::CommandExit(code) => {
                let c = code.map(|c| c.to_string()).unwrap_or_else(|| "?".into());
                s.err_command_exit.replace("{code}", &c)
            }
            Kind::SttScriptMissing => s.err_stt_script.to_string(),
            Kind::NoMonitors => s.err_no_monitors.to_string(),
            Kind::CaptureFailed => s.err_capture.to_string(),
        };
        let detail = self.detail.trim();
        if detail.is_empty() {
            head
        } else if matches!(self.kind, Kind::CommandExit(_)) {
            format!("{head}\n```\n{detail}\n```")
        } else {
            format!("{head}: {detail}")
        }
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text(Lang::En))
    }
}

impl std::error::Error for AppError {}

/// Localizes an error from the backend; errors that are not `AppError` (API messages, I/O) stay raw.
pub fn localize(e: &anyhow::Error, lang: Lang) -> String {
    match e.downcast_ref::<AppError>() {
        Some(a) => a.text(lang),
        None => e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_and_keeps_detail() {
        let e = AppError::with(Kind::RunFailed("python".into()), "os error 2");
        assert_eq!(localize(&e, Lang::En), "Could not run python: os error 2");
        assert!(localize(&e, Lang::Ru).starts_with("Не удалось запустить python"));
        assert!(localize(&e, Lang::Uk).ends_with("os error 2"));
        let raw = anyhow::anyhow!("HTTP 401");
        assert_eq!(localize(&raw, Lang::Uk), "HTTP 401");
    }
}
