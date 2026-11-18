//! Runs the faster-whisper Python sidecar and feeds it audio segments.

use crate::events::Event;
use crossbeam_channel::{Receiver, Sender};
use std::io::{BufRead, BufReader, Write};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

#[cfg(windows)]
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub struct Stt {
    child: Child,
}

impl Drop for Stt {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

fn find_script() -> Option<PathBuf> {
    let name = "stt_server.py";
    let mut cands = vec![];
    if let Ok(exe) = std::env::current_exe() {
        let mut d = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..4 {
            if let Some(dir) = d {
                cands.push(dir.join(name));
                d = dir.parent().map(|p| p.to_path_buf());
            } else {
                break;
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        cands.push(cwd.join(name));
    }
    cands.into_iter().find(|p| p.exists())
}

pub fn start(
    python: &str,
    model: &str,
    language: &str,
    device: &str,
    prompt: &str,
    segments: Receiver<Vec<f32>>,
    ev: Sender<Event>,
    ctx: eframe::egui::Context,
) -> anyhow::Result<Stt> {
    let script = find_script().ok_or_else(|| crate::errors::AppError::new(crate::errors::Kind::SttScriptMissing))?;
    let mut cmd = Command::new(python);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let mut child = cmd
        .arg("-u")
        .arg(&script)
        .args(["--model", model, "--language", language, "--device", device, "--prompt", prompt])
        .env("PYTHONIOENCODING", "utf-8")
        .env("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| crate::errors::AppError::with(crate::errors::Kind::RunFailed(python.to_string()), e))?;

    let mut stdin = child.stdin.take().unwrap();
    std::thread::spawn(move || {
        for seg in segments {
            let mut buf = Vec::with_capacity(4 + seg.len() * 4);
            buf.extend_from_slice(&(seg.len() as u32).to_le_bytes());
            for s in seg {
                buf.extend_from_slice(&s.to_le_bytes());
            }
            if stdin.write_all(&buf).and_then(|_| stdin.flush()).is_err() {
                break;
            }
        }
    });

    let stdout = child.stdout.take().unwrap();
    let (ev2, ctx2) = (ev.clone(), ctx.clone());
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            let s = |k: &str| v[k].as_str().unwrap_or_default().to_string();
            let e = match v["type"].as_str() {
                Some("text") => Event::Transcript(s("text")),
                Some("ready") => Event::SttReady {
                    model: s("model"),
                    device: s("device").to_uppercase(),
                    load_s: v["load_s"].to_string(),
                },
                _ => Event::Log(format!("stt: {}", s("text"))),
            };
            let _ = ev2.send(e);
            ctx2.request_repaint();
        }
        let _ = ev2.send(Event::SttStopped);
        ctx2.request_repaint();
    });

    let stderr = child.stderr.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let l = line.trim();
            if l.contains("Error") || l.contains("error") || l.starts_with("Traceback") {
                let _ = ev.send(Event::Log(format!("stt stderr: {l}")));
                ctx.request_repaint();
            }
        }
    });

    Ok(Stt { child })
}
