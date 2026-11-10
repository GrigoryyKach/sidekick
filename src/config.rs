use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_SYSTEM_PROMPT: &str = "\
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

/// Nudges Whisper to keep English tech terms in Latin script.
pub const DEFAULT_STT_PROMPT: &str = "Технический созвон. Термины пишем латиницей: React, useEffect, useState, JavaScript, TypeScript, Node.js, Python, race condition, deadlock, mutex, API, REST, SQL, Docker, Kubernetes, Git.";

/// Кто отвечает на вопросы.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    /// Claude Code CLI (`claude -p`) на подписке пользователя.
    #[default]
    ClaudeCode,
    /// Anthropic Messages API по ключу.
    Anthropic,
    /// Любой OpenAI-совместимый сервер: OpenAI, OpenRouter, Ollama, LM Studio, vLLM…
    OpenAi,
    /// Своя консольная команда: вопрос приходит в stdin, ответ читается из stdout.
    Command,
}

impl AgentKind {
    pub const ALL: [AgentKind; 4] = [AgentKind::ClaudeCode, AgentKind::Anthropic, AgentKind::OpenAi, AgentKind::Command];

    pub fn title(self) -> &'static str {
        match self {
            AgentKind::ClaudeCode => "Claude Code",
            AgentKind::Anthropic => "Anthropic API",
            AgentKind::OpenAi => "OpenAI-совместимый",
            AgentKind::Command => "Своя команда",
        }
    }
}

pub const DEFAULT_ANTHROPIC_URL: &str = "https://api.anthropic.com";
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-opus-5-5";
pub const DEFAULT_OPENAI_URL: &str = "https://api.openai.com/v1";

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub monitor_index: usize,
    pub opacity: f32,
    pub font_size: f32,
    pub model: String,
    pub effort: String,
    pub claude_path: String,
    pub agent: AgentKind,
    /// Адрес API (пусто — адрес по умолчанию для выбранного агента).
    pub api_base_url: String,
    /// Ключ API (пусто — берётся из ANTHROPIC_API_KEY / OPENAI_API_KEY).
    pub api_key: String,
    /// Модель для Anthropic API / OpenAI-совместимого сервера.
    pub api_model: String,
    /// Команда для агента «Своя команда». `{image}` заменяется путём к скриншоту.
    pub command_line: String,
    /// Проверять обновления при запуске (NIMVORO).
    pub check_updates: bool,
    pub python_path: String,
    pub whisper_model: String,
    pub whisper_device: String,
    pub stt_language: String,
    pub stt_prompt: String,
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
            stt_language: "ru".into(),
            stt_prompt: DEFAULT_STT_PROMPT.into(),
            answer_language: "русском".into(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
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

pub fn app_dir() -> PathBuf {
    let d = dirs::config_dir().unwrap_or_else(std::env::temp_dir).join("transluent");
    let _ = std::fs::create_dir_all(&d);
    d
}

fn config_path() -> PathBuf {
    app_dir().join("config.json")
}

impl Config {
    pub fn load() -> Self {
        std::fs::read_to_string(config_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let Ok(s) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(config_path(), s);
        }
    }

    /// Модель, которую видит пользователь в строке состояния.
    pub fn agent_model(&self) -> String {
        match self.agent {
            AgentKind::ClaudeCode => self.model.clone(),
            AgentKind::Anthropic if self.api_model.trim().is_empty() => DEFAULT_ANTHROPIC_MODEL.into(),
            AgentKind::Anthropic | AgentKind::OpenAi => self.api_model.trim().to_string(),
            AgentKind::Command => self.command_line.split_whitespace().next().unwrap_or("команда").to_string(),
        }
    }

    pub fn full_system_prompt(&self) -> String {
        format!(
            "{}\n\nОтвечай на {} языке (код и термины — как принято).",
            self.system_prompt, self.answer_language
        )
    }
}
