//! Агенты, которые отвечают на вопросы. Все шлют одни и те же события
//! (`AgentReady` / `AgentDelta` / `AgentDone` / `AgentExited`), поэтому интерфейсу всё равно, кто отвечает.
//!
//! - Claude Code — постоянный процесс `claude -p` (см. `claude.rs`).
//! - Anthropic API — Messages API по ключу, потоковый ответ (SSE).
//! - OpenAI-совместимый — `/chat/completions` с `stream: true` (OpenAI, OpenRouter, Ollama, LM Studio, vLLM…).
//! - Своя команда — на каждый вопрос запускается команда, вопрос уходит в stdin, stdout показывается как ответ.

use crate::config::{AgentKind, Config, DEFAULT_ANTHROPIC_MODEL, DEFAULT_ANTHROPIC_URL, DEFAULT_OPENAI_URL, app_dir};
use crate::events::Event;
use crate::stt::CREATE_NO_WINDOW;
use base64::Engine;
use crossbeam_channel::{Sender, unbounded};
use eframe::egui;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::windows::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::Duration;

pub trait Agent: Send {
    /// `false` — агент умер и его нужно перезапустить.
    fn alive(&mut self) -> bool;
    /// Отправить вопрос. Ответ придёт событиями.
    fn send(&mut self, text: &str, image_png_b64: Option<&str>) -> anyhow::Result<()>;
}

pub fn start(cfg: &Config, ev: Sender<Event>, ctx: egui::Context) -> anyhow::Result<Box<dyn Agent>> {
    match cfg.agent {
        AgentKind::ClaudeCode => Ok(Box::new(crate::claude::Claude::start(cfg, ev, ctx)?)),
        AgentKind::Anthropic | AgentKind::OpenAi | AgentKind::Command => {
            let backend = Backend::from_config(cfg)?;
            Ok(Box::new(Worker::spawn(backend, cfg.full_system_prompt(), ev, ctx)))
        }
    }
}

// ───────────────────────── общий фоновый исполнитель ─────────────────────────

/// Вопросы выполняются по очереди в отдельном потоке, чтобы ответы не перемешивались.
struct Worker {
    jobs: Sender<(String, Option<String>)>,
}

impl Worker {
    fn spawn(backend: Backend, system: String, ev: Sender<Event>, ctx: egui::Context) -> Self {
        let (jobs, rx) = unbounded::<(String, Option<String>)>();
        std::thread::spawn(move || {
            let mut history: Vec<Value> = Vec::new();
            let _ = ev.send(Event::AgentReady);
            ctx.request_repaint();
            for (text, img) in rx {
                let emit = |e: Event| {
                    let _ = ev.send(e);
                    ctx.request_repaint();
                };
                let res = match &backend {
                    Backend::Anthropic(b) => b.ask_anthropic(&system, &mut history, &text, img.as_deref(), &emit),
                    Backend::OpenAi(b) => b.ask_openai(&system, &mut history, &text, img.as_deref(), &emit),
                    Backend::Command(line) => ask_command(line, &system, &text, img.as_deref(), &emit),
                };
                match res {
                    Ok(full) => emit(Event::AgentDone { text: full, is_error: false }),
                    Err(e) => emit(Event::AgentDone { text: format!("Ошибка: {e}"), is_error: true }),
                }
            }
        });
        Self { jobs }
    }
}

impl Agent for Worker {
    fn alive(&mut self) -> bool {
        true
    }

    fn send(&mut self, text: &str, image_png_b64: Option<&str>) -> anyhow::Result<()> {
        self.jobs
            .send((text.to_string(), image_png_b64.map(str::to_string)))
            .map_err(|_| anyhow::anyhow!("агент остановлен"))
    }
}

enum Backend {
    Anthropic(Http),
    OpenAi(Http),
    Command(String),
}

struct Http {
    base: String,
    key: String,
    model: String,
    effort: String,
    agent: ureq::Agent,
}

impl Backend {
    fn from_config(cfg: &Config) -> anyhow::Result<Self> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_global(Some(Duration::from_secs(600)))
            .build()
            .into();
        let base = |default: &str| {
            let b = cfg.api_base_url.trim();
            if b.is_empty() { default.to_string() } else { b.trim_end_matches('/').to_string() }
        };
        let key = |env: &str| {
            let k = cfg.api_key.trim();
            if k.is_empty() { std::env::var(env).unwrap_or_default() } else { k.to_string() }
        };
        match cfg.agent {
            AgentKind::Anthropic => {
                let key = key("ANTHROPIC_API_KEY");
                if key.is_empty() {
                    anyhow::bail!("нет ключа Anthropic API: укажите его в настройках или в ANTHROPIC_API_KEY");
                }
                let model = if cfg.api_model.trim().is_empty() { DEFAULT_ANTHROPIC_MODEL.into() } else { cfg.api_model.trim().into() };
                Ok(Backend::Anthropic(Http { base: base(DEFAULT_ANTHROPIC_URL), key, model, effort: cfg.effort.trim().into(), agent }))
            }
            AgentKind::OpenAi => {
                if cfg.api_model.trim().is_empty() {
                    anyhow::bail!("укажите модель в настройках агента");
                }
                // Ключ необязателен: локальным Ollama / LM Studio он не нужен.
                Ok(Backend::OpenAi(Http {
                    base: base(DEFAULT_OPENAI_URL),
                    key: key("OPENAI_API_KEY"),
                    model: cfg.api_model.trim().into(),
                    effort: String::new(),
                    agent,
                }))
            }
            AgentKind::Command => {
                if cfg.command_line.trim().is_empty() {
                    anyhow::bail!("укажите команду агента в настройках");
                }
                Ok(Backend::Command(cfg.command_line.trim().into()))
            }
            AgentKind::ClaudeCode => unreachable!(),
        }
    }
}

/// Читает Server-Sent Events: для каждого `data:` вызывает `on_data`. `false` из `on_data` — остановиться.
fn read_sse(body: impl Read, mut on_data: impl FnMut(&str) -> anyhow::Result<bool>) -> anyhow::Result<()> {
    for line in BufReader::new(body).lines() {
        let line = line?;
        if let Some(data) = line.strip_prefix("data:") {
            if !on_data(data.trim())? {
                break;
            }
        }
    }
    Ok(())
}

fn http_error(status: u16, body: &str) -> anyhow::Error {
    let msg = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string).or_else(|| v["error"].as_str().map(str::to_string)))
        .unwrap_or_else(|| body.chars().take(300).collect());
    anyhow::anyhow!("HTTP {status}: {msg}")
}

// ───────────────────────── Anthropic Messages API ─────────────────────────

impl Http {
    /// Anthropic: история хранится целиком, вместе с блоками размышлений, как пришла от API.
    fn ask_anthropic(
        &self,
        system: &str,
        history: &mut Vec<Value>,
        text: &str,
        img: Option<&str>,
        emit: &dyn Fn(Event),
    ) -> anyhow::Result<String> {
        let mut content = vec![];
        if let Some(b64) = img {
            content.push(json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}));
        }
        content.push(json!({"type": "text", "text": text}));
        history.push(json!({"role": "user", "content": content}));

        let mut body = json!({
            "model": self.model,
            "max_tokens": 64000,
            "stream": true,
            "system": system,
            "messages": history,
        });
        if !self.effort.is_empty() {
            body["output_config"] = json!({"effort": self.effort});
        }
        // Если модель откажется отвечать по соображениям безопасности, API сам повторит запрос на запасной модели.
        let fallback = self.model.starts_with("claude-");
        if fallback {
            body["fallbacks"] = json!("default");
        }
        let mut req = self
            .agent
            .post(format!("{}/v1/messages", self.base))
            .header("x-api-key", &self.key)
            .header("anthropic-version", "2026-10-21")
            .header("content-type", "application/json");
        if fallback {
            req = req.header("anthropic-beta", "server-side-fallback-2026-07-01");
        }
        let resp = req.send(body.to_string().as_str());
        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                history.pop();
                return Err(e.into());
            }
        };
        let status = resp.status().as_u16();
        let mut reader = resp.into_body().into_reader();
        if status != 200 {
            history.pop();
            let mut s = String::new();
            let _ = reader.read_to_string(&mut s);
            return Err(http_error(status, &s));
        }

        // Собираем блоки ответа из дельт, чтобы вернуть их в историю без изменений.
        let mut blocks: Vec<Value> = vec![];
        let mut full = String::new();
        let mut stop_reason = String::new();
        let res = read_sse(reader, |data| {
            let v: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => return Ok(true),
            };
            match v["type"].as_str().unwrap_or_default() {
                "content_block_start" => {
                    let i = v["index"].as_u64().unwrap_or(0) as usize;
                    if blocks.len() <= i {
                        blocks.resize(i + 1, Value::Null);
                    }
                    blocks[i] = v["content_block"].clone();
                }
                "content_block_delta" => {
                    let i = v["index"].as_u64().unwrap_or(0) as usize;
                    let d = &v["delta"];
                    let Some(b) = blocks.get_mut(i) else { return Ok(true) };
                    match d["type"].as_str().unwrap_or_default() {
                        "text_delta" => {
                            let t = d["text"].as_str().unwrap_or_default();
                            full += t;
                            append_str(b, "text", t);
                            emit(Event::AgentDelta(t.to_string()));
                        }
                        "thinking_delta" => append_str(b, "thinking", d["thinking"].as_str().unwrap_or_default()),
                        "signature_delta" => b["signature"] = d["signature"].clone(),
                        "citations_delta" => {}
                        _ => {}
                    }
                }
                "message_delta" => {
                    if let Some(s) = v["delta"]["stop_reason"].as_str() {
                        stop_reason = s.to_string();
                    }
                }
                "error" => anyhow::bail!("{}", v["error"]["message"].as_str().unwrap_or("ошибка API")),
                "message_stop" => return Ok(false),
                _ => {}
            }
            Ok(true)
        });
        if let Err(e) = res {
            history.pop();
            return Err(e);
        }
        let blocks: Vec<Value> = blocks.into_iter().filter(|b| !b.is_null()).collect();
        if !blocks.is_empty() {
            history.push(json!({"role": "assistant", "content": blocks}));
        } else {
            history.pop();
        }
        if stop_reason == "refusal" {
            anyhow::bail!("модель отказалась отвечать на этот запрос");
        }
        Ok(full)
    }

    /// OpenAI-совместимый `/chat/completions`. В историю идёт только текст.
    fn ask_openai(
        &self,
        system: &str,
        history: &mut Vec<Value>,
        text: &str,
        img: Option<&str>,
        emit: &dyn Fn(Event),
    ) -> anyhow::Result<String> {
        let user = match img {
            Some(b64) => json!({"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}},
            ]}),
            None => json!({"role": "user", "content": text}),
        };
        history.push(user);
        let mut messages = vec![json!({"role": "system", "content": system})];
        messages.extend(history.iter().cloned());
        let body = json!({"model": self.model, "stream": true, "messages": messages});

        let mut req = self.agent.post(format!("{}/chat/completions", self.base)).header("content-type", "application/json");
        if !self.key.is_empty() {
            req = req.header("authorization", &format!("Bearer {}", self.key));
        }
        let resp = match req.send(body.to_string().as_str()) {
            Ok(r) => r,
            Err(e) => {
                history.pop();
                return Err(e.into());
            }
        };
        let status = resp.status().as_u16();
        let mut reader = resp.into_body().into_reader();
        if status != 200 {
            history.pop();
            let mut s = String::new();
            let _ = reader.read_to_string(&mut s);
            return Err(http_error(status, &s));
        }
        let mut full = String::new();
        let res = read_sse(reader, |data| {
            if data == "[DONE]" {
                return Ok(false);
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else { return Ok(true) };
            if let Some(msg) = v["error"]["message"].as_str() {
                anyhow::bail!("{msg}");
            }
            if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
                if !t.is_empty() {
                    full += t;
                    emit(Event::AgentDelta(t.to_string()));
                }
            }
            Ok(true)
        });
        if let Err(e) = res {
            history.pop();
            return Err(e);
        }
        history.push(json!({"role": "assistant", "content": full}));
        Ok(full)
    }
}

fn append_str(block: &mut Value, key: &str, add: &str) {
    let cur = block[key].as_str().unwrap_or_default().to_string();
    block[key] = Value::String(cur + add);
}

// ───────────────────────── своя команда ─────────────────────────

/// Запускает команду через `cmd /C`, пишет в stdin системный промпт и вопрос, читает stdout как ответ.
/// Если в команде есть `{image}`, туда подставляется путь к PNG со скриншотом.
fn ask_command(line: &str, system: &str, text: &str, img: Option<&str>, emit: &dyn Fn(Event)) -> anyhow::Result<String> {
    let mut line = line.to_string();
    if line.contains("{image}") {
        let path = match img {
            Some(b64) => {
                let p = app_dir().join("last_screenshot.png");
                std::fs::write(&p, base64::engine::general_purpose::STANDARD.decode(b64)?)?;
                format!("\"{}\"", p.display())
            }
            None => String::new(),
        };
        line = line.replace("{image}", &path);
    }
    let mut child = Command::new("cmd")
        .arg("/C")
        .raw_arg(&line)
        .current_dir(app_dir())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map_err(|e| anyhow::anyhow!("запуск «{line}»: {e}"))?;
    {
        let mut stdin = child.stdin.take().unwrap();
        let prompt = format!("{system}\n\n{text}\n");
        stdin.write_all(prompt.as_bytes())?;
    }
    let mut stderr = child.stderr.take().unwrap();
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
    let mut out = child.stdout.take().unwrap();
    let mut full = String::new();
    let mut buf = [0u8; 4096];
    let mut pending: Vec<u8> = vec![];
    loop {
        let n = out.read(&mut buf)?;
        if n == 0 {
            break;
        }
        pending.extend_from_slice(&buf[..n]);
        // Отдаём только целые UTF-8 символы.
        let valid = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid > 0 {
            let chunk = String::from_utf8_lossy(&pending[..valid]).into_owned();
            pending.drain(..valid);
            full += &chunk;
            emit(Event::AgentDelta(chunk));
        }
    }
    let status = child.wait()?;
    let err = err_thread.join().unwrap_or_default();
    if !status.success() && full.trim().is_empty() {
        anyhow::bail!("команда завершилась с кодом {:?}\n```\n{}\n```", status.code(), err.trim());
    }
    Ok(full)
}
