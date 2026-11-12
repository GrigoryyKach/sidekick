use crate::i18n::{self, Lang};
use crate::theme::ThemeMode;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const SYSTEM_PROMPT_EN: &str = "\
You are an assistant that helps the user in real time during a call or a live-coding session.
You receive: (a) a transcript of what the other participants say (this is NOT the user but the other people on the call), \
(b) sometimes a screenshot of the user's screen, (c) sometimes a direct question from the user.

Rules:
- Answer right away and to the point, without a preamble and without restating the question.
- Start with a short gist (1-3 lines) that can be said out loud immediately.
- If the screen shows a programming task: detect the language from the screen/editor, briefly describe the idea, \
then give complete working code in a ``` block, then the time and space complexity. Respect any function signature \
that is already on the screen.
- If it is a question from another participant (theory, system design, behavioural), give talking points that are easy to retell in your own words.
- If the transcript is fragmentary or has recognition errors, guess what it is about from the context.
- Format: Markdown, brief. Do not use tools.";

/// Nudges Whisper to keep English tech terms in Latin script.
const STT_PROMPT_EN: &str = "Technical call. Keep terms in Latin script: React, useEffect, useState, JavaScript, TypeScript, Node.js, Python, race condition, deadlock, mutex, API, REST, SQL, Docker, Kubernetes, Git.";

const SYSTEM_PROMPT_RU: &str = "\
Ты — ассистент, который в реальном времени помогает пользователю на созвоне или во время live coding.
Тебе приходят: (а) расшифровка речи собеседников (это НЕ пользователь, а другие участники звонка), \
(б) иногда скриншот экрана пользователя, (в) иногда прямой вопрос пользователя.

Правила ответа:
- Отвечай сразу по делу, без вступлений и без пересказа вопроса.
- Начни с короткой сути (1–3 строки), которую можно сразу произнести вслух.
- Если на экране задача на программирование: определи язык по экрану/редактору, кратко опиши идею, \
затем дай полный рабочий код в блоке ```, затем сложность по времени и памяти. Учитывай сигнатуру функции, \
которая уже есть на экране.
- Если это вопрос собеседника (теория, system design, поведенческий) — дай тезисы, которые легко пересказать своими словами.
- Если расшифровка обрывочная или с ошибками распознавания — догадайся по контексту, о чём речь.
- Формат — Markdown, коротко. Не используй инструменты.";
const STT_PROMPT_RU: &str = "Технический созвон. Термины пишем латиницей: React, useEffect, useState, JavaScript, TypeScript, Node.js, Python, race condition, deadlock, mutex, API, REST, SQL, Docker, Kubernetes, Git.";

const SYSTEM_PROMPT_UK: &str = "Ти — асистент, який у реальному часі допомагає користувачеві на дзвінку чи під час live coding.
Тобі надходять: (а) розшифровка мовлення співрозмовників (це НЕ користувач, а інші учасники дзвінка), (б) інколи знімок екрана користувача, (в) інколи пряме запитання користувача.

Правила відповіді:
- Відповідай одразу по суті, без вступів і без переказу запитання.
- Почни з короткої суті (1–3 рядки), яку можна відразу промовити вголос.
- Якщо на екрані задача з програмування: визнач мову за екраном/редактором, коротко опиши ідею, потім дай повний робочий код у блоці ```, потім складність за часом і пам'яттю. Враховуй сигнатуру функції, яка вже є на екрані.
- Якщо це запитання співрозмовника (теорія, system design, поведінкове) — дай тези, які легко переказати своїми словами.
- Якщо розшифровка уривчаста або з помилками розпізнавання — здогадайся за контекстом, про що йдеться.
- Формат — Markdown, коротко. Не використовуй інструменти.";
const STT_PROMPT_UK: &str = "Технічний дзвінок. Терміни пишемо латиницею: React, useEffect, useState, JavaScript, TypeScript, Node.js, Python, race condition, deadlock, mutex, API, REST, SQL, Docker, Kubernetes, Git.";

/// Default system prompt for a fresh config, in the interface language.
pub fn default_system_prompt(lang: Lang) -> &'static str {
    match lang {
        Lang::En => SYSTEM_PROMPT_EN,
        Lang::Ru => SYSTEM_PROMPT_RU,
        Lang::Uk => SYSTEM_PROMPT_UK,
    }
}

/// Default Whisper hint for a fresh config, in the interface language.
pub fn default_stt_prompt(lang: Lang) -> &'static str {
    match lang {
        Lang::En => STT_PROMPT_EN,
        Lang::Ru => STT_PROMPT_RU,
        Lang::Uk => STT_PROMPT_UK,
    }
}

/// Who answers the questions.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    /// Claude Code CLI (`claude -p`) on the user's subscription.
    #[default]
    ClaudeCode,
    /// Anthropic Messages API with a key.
    Anthropic,
    /// Any OpenAI-compatible server: OpenAI, OpenRouter, Ollama, LM Studio, vLLM...
    OpenAi,
    /// A custom console command: the question goes to stdin, the answer is read from stdout.
    Command,
}

impl AgentKind {
    pub const ALL: [AgentKind; 4] = [AgentKind::ClaudeCode, AgentKind::Anthropic, AgentKind::OpenAi, AgentKind::Command];

    /// Position in `AgentKind::ALL` (and in the localized title/description arrays).
    pub fn index(self) -> usize {
        match self {
            AgentKind::ClaudeCode => 0,
            AgentKind::Anthropic => 1,
            AgentKind::OpenAi => 2,
            AgentKind::Command => 3,
        }
    }
}

pub const DEFAULT_ANTHROPIC_URL: &str = "https://api.anthropic.com";
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-opus-5-5";
pub const DEFAULT_OPENAI_URL: &str = "https://api.openai.com/v1";

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Interface language code (`en`, `ru`, `uk`); `None` until first resolved.
    pub language: Option<String>,
    /// `system`, `light` or `dark`.
    pub theme: ThemeMode,
    pub monitor_index: usize,
    pub opacity: f32,
    pub font_size: f32,
    pub model: String,
    pub effort: String,
    pub claude_path: String,
    pub agent: AgentKind,
    /// API address (empty: the default for the selected agent).
    pub api_base_url: String,
    /// API key (empty: taken from ANTHROPIC_API_KEY / OPENAI_API_KEY).
    pub api_key: String,
    /// Model for the Anthropic API / OpenAI-compatible server.
    pub api_model: String,
    /// Command for the "custom command" agent. `{image}` is replaced with the screenshot path.
    pub command_line: String,
    /// Check for updates on startup (NIMVORO).
    pub check_updates: bool,
    pub python_path: String,
    pub whisper_model: String,
    pub whisper_device: String,
    pub stt_language: String,
    pub stt_prompt: String,
    /// Language of the answers; empty means "same as the interface language".
    pub answer_language: String,
    pub system_prompt: String,
    pub vad_threshold: f32,
    pub silence_ms: u64,
    pub max_segment_s: f32,
    pub max_image_edge: u32,
    pub listen_on_start: bool,
    pub window_pos: Option<[f32; 2]>,
    pub window_size: [f32; 2],
}

impl Default for Config {
    fn default() -> Self {
        Self {
            language: None,
            theme: ThemeMode::System,
            monitor_index: 0,
            opacity: 0.80,
            font_size: 14.0,
            model: "sonnet".into(),
            effort: String::new(),
            claude_path: String::new(),
            agent: AgentKind::ClaudeCode,
            api_base_url: String::new(),
            api_key: String::new(),
            api_model: String::new(),
            command_line: String::new(),
            check_updates: true,
            python_path: "python".into(),
            whisper_model: "large-v3-turbo".into(),
            whisper_device: "cuda".into(),
            stt_language: "auto".into(),
            stt_prompt: default_stt_prompt(i18n::resolve(None)).into(),
            answer_language: String::new(),
            system_prompt: default_system_prompt(i18n::resolve(None)).into(),
            vad_threshold: 0.006,
            silence_ms: 800,
            max_segment_s: 20.0,
            max_image_edge: 1800,
            listen_on_start: true,
            window_pos: None,
            window_size: [560.0, 680.0],
        }
    }
}

/// Folder with the settings and the log: `%APPDATA%\Sidekick`.
/// `SIDEKICK_DATA_DIR` overrides it (portable runs, tests); nothing is migrated then.
/// On the first start after the rename from "transluent" the old settings are copied over.
pub fn app_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("SIDEKICK_DATA_DIR").filter(|d| !d.is_empty()) {
        let d = PathBuf::from(d);
        let _ = std::fs::create_dir_all(&d);
        return d;
    }
    let base = dirs::config_dir().unwrap_or_else(std::env::temp_dir);
    let d = base.join("Sidekick");
    migrate_legacy(&base.join("transluent"), &d);
    let _ = std::fs::create_dir_all(&d);
    d
}

/// Copies `config.json` from the old settings folder when the new folder does not exist yet.
/// The old folder is left untouched. Returns `true` when something was copied.
pub fn migrate_legacy(old: &Path, new: &Path) -> bool {
    if new.exists() || !old.join("config.json").is_file() {
        return false;
    }
    if std::fs::create_dir_all(new).is_err() {
        return false;
    }
    std::fs::copy(old.join("config.json"), new.join("config.json")).is_ok()
}

fn config_path() -> PathBuf {
    app_dir().join("config.json")
}

impl Config {
    pub fn load() -> Self {
        let path = config_path();
        let Ok(text) = std::fs::read_to_string(&path) else { return Self::default() };
        match serde_json::from_str::<Config>(text.trim_start_matches('\u{feff}')) {
            // Saved prompts are never touched: existing installs keep behaving as before.
            Ok(c) => c,
            Err(_) => {
                // Keep an unreadable file instead of overwriting it with defaults on the next save.
                let _ = std::fs::copy(&path, path.with_extension("invalid.json"));
                Self::default()
            }
        }
    }

    pub fn save(&self) {
        if let Ok(s) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(config_path(), s);
        }
    }

    /// Interface language (the config value, then the NIMVORO marker, then English).
    pub fn lang(&self) -> Lang {
        i18n::resolve(self.language.as_deref())
    }

    /// Model shown to the user in the status line.
    pub fn agent_model(&self) -> String {
        match self.agent {
            AgentKind::ClaudeCode => self.model.clone(),
            AgentKind::Anthropic if self.api_model.trim().is_empty() => DEFAULT_ANTHROPIC_MODEL.into(),
            AgentKind::Anthropic | AgentKind::OpenAi => self.api_model.trim().to_string(),
            AgentKind::Command => self.command_line.split_whitespace().next().unwrap_or("command").to_string(),
        }
    }

    pub fn full_system_prompt(&self) -> String {
        let answer_language = match self.answer_language.trim() {
            "" => self.lang().pick("English", "Russian", "Ukrainian"),
            custom => custom,
        };
        format!("{}\n\nReply in {} (keep code and technical terms as they are normally written).", self.system_prompt, answer_language)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_old_config_once() {
        let root = std::env::temp_dir().join(format!("sidekick-test-{}", std::process::id()));
        let (old, new) = (root.join("transluent"), root.join("Sidekick"));
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("config.json"), "{\"opacity\":0.5}").unwrap();
        assert!(migrate_legacy(&old, &new));
        assert_eq!(std::fs::read_to_string(new.join("config.json")).unwrap(), "{\"opacity\":0.5}");
        // The new folder exists now: the old file must not overwrite later changes.
        std::fs::write(new.join("config.json"), "{}").unwrap();
        assert!(!migrate_legacy(&old, &new));
        assert_eq!(std::fs::read_to_string(new.join("config.json")).unwrap(), "{}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn old_config_still_parses() {
        let c: Config = serde_json::from_str("{\"agent\":\"anthropic\",\"opacity\":0.5}").unwrap();
        assert_eq!(c.agent, AgentKind::Anthropic);
        assert_eq!(c.theme, ThemeMode::System);
        assert!(c.language.is_none());
    }
}
