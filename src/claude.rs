//! Persistent `claude -p` (Claude Code, headless) session speaking stream-json.
//! Auth is whatever Claude Code itself is logged in with (your subscription via browser).

use crate::config::{Config, app_dir};
use crate::events::Event;
#[cfg(windows)]
use crate::stt::CREATE_NO_WINDOW;
use crossbeam_channel::Sender;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};

pub struct Claude {
    child: Child,
    stdin: ChildStdin,
}

impl Drop for Claude {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

pub fn resolve_claude(custom: &str) -> Option<PathBuf> {
    if !custom.trim().is_empty() {
        return Some(PathBuf::from(custom.trim()));
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        #[cfg(windows)]
        {
            // npm install: claude.cmd is a wrapper around the native binary.
            let native = dir.join("node_modules/@anthropic-ai/claude-code/bin/claude.exe");
            if dir.join("claude.cmd").exists() && native.exists() {
                return Some(native);
            }
            let exe = dir.join("claude.exe");
            if exe.exists() {
                return Some(exe);
            }
        }
        #[cfg(not(windows))]
        {
            let exe = dir.join("claude");
            if exe.is_file() {
                return Some(exe);
            }
        }
    }
    #[cfg(windows)]
    let local = dirs::home_dir()?.join(".local/bin/claude.exe");
    #[cfg(not(windows))]
    let local = dirs::home_dir()?.join(".local/bin/claude");
    local.exists().then_some(local)
}

impl Claude {
    pub fn start(cfg: &Config, ev: Sender<Event>, ctx: eframe::egui::Context) -> anyhow::Result<Self> {
        let exe = resolve_claude(&cfg.claude_path)
            .ok_or_else(|| crate::errors::AppError::new(crate::errors::Kind::ClaudeNotFound))?;
        let dir = app_dir().join("claude_cwd");
        let _ = std::fs::create_dir_all(&dir);
        let prompt_file = app_dir().join("system_prompt.txt");
        std::fs::write(&prompt_file, cfg.full_system_prompt())?;

        let mut cmd = Command::new(&exe);
        cmd.args([
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--tools",
            "",
            "--model",
            &cfg.model,
        ])
        .arg("--system-prompt-file")
        .arg(&prompt_file);
        if !cfg.effort.trim().is_empty() {
            cmd.args(["--effort", cfg.effort.trim()]);
        }
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);
        let mut child = cmd
            .current_dir(&dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| crate::errors::AppError::with(crate::errors::Kind::RunFailed(exe.display().to_string()), e))?;
        let stdin = child.stdin.take().unwrap();

        let stdout = child.stdout.take().unwrap();
        let (ev2, ctx2) = (ev.clone(), ctx.clone());
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
                if let Some(e) = parse(&v) {
                    let _ = ev2.send(e);
                    ctx2.request_repaint();
                }
            }
        });

        let stderr = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            let mut tail: Vec<String> = vec![];
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                tail.push(line);
                if tail.len() > 6 {
                    tail.remove(0);
                }
            }
            let _ = ev.send(Event::AgentExited(tail.join("\n")));
            ctx.request_repaint();
        });

        Ok(Self { child, stdin })
    }

}

impl crate::agent::Agent for Claude {
    fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn send(&mut self, text: &str, image_png_b64: Option<&str>) -> anyhow::Result<()> {
        let mut content = vec![];
        if let Some(img) = image_png_b64 {
            content.push(json!({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img}
            }));
        }
        content.push(json!({"type": "text", "text": text}));
        let msg = json!({"type": "user", "message": {"role": "user", "content": content}});
        let mut line = serde_json::to_string(&msg)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.flush()?;
        Ok(())
    }
}

fn parse(v: &Value) -> Option<Event> {
    match v["type"].as_str()? {
        "system" if v["subtype"] == "init" => Some(Event::AgentReady),
        "stream_event" => {
            let e = &v["event"];
            if e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta" {
                Some(Event::AgentDelta(e["delta"]["text"].as_str()?.to_string()))
            } else {
                None
            }
        }
        "result" => Some(Event::AgentDone {
            text: v["result"].as_str().unwrap_or_default().to_string(),
            is_error: v["is_error"].as_bool().unwrap_or(false) || v["subtype"] != "success",
        }),
        _ => None,
    }
}
