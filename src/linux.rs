//! Linux counterpart of `win.rs` with the same public interface: show/hide and global hotkeys.
//!
//! There is no Linux equivalent of `SetWindowDisplayAffinity`: the window cannot be hidden from screen sharing,
//! so `is_capture_excluded` is always `false` and the interface says so. Global hotkeys work on X11 through the
//! `global-hotkey` crate; Wayland gives programs no global hotkeys, so there everything is done from the window.

use crate::events::{Event, Hotkey};
use crossbeam_channel::Sender;
use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

static CTX: OnceLock<eframe::egui::Context> = OnceLock::new();
static VISIBLE: AtomicBool = AtomicBool::new(true);

/// The window exists as soon as the first frame runs; there are no extra styles to apply.
pub fn init_window(_title: &str) -> bool {
    true
}

/// Nothing to re-apply on Linux (the Windows build restores its tool-window style here).
pub fn ensure_tool_window() {}

/// Hiding from screen capture is not possible on Linux.
pub fn is_capture_excluded() -> bool {
    false
}

pub fn is_visible() -> bool {
    VISIBLE.load(Ordering::Relaxed)
}

pub fn set_visible(v: bool) {
    VISIBLE.store(v, Ordering::Relaxed);
    if let Some(ctx) = CTX.get() {
        ctx.send_viewport_cmd(eframe::egui::ViewportCommand::Visible(v));
        ctx.request_repaint();
    }
}

const KEYS: &[(Code, Hotkey)] = &[
    (Code::Space, Hotkey::AskScreen),
    (Code::Enter, Hotkey::AskTranscript),
    (Code::KeyH, Hotkey::ToggleVisible),
    (Code::KeyT, Hotkey::ToggleClickThrough),
    (Code::KeyL, Hotkey::ToggleListen),
    (Code::KeyN, Hotkey::NewChat),
    (Code::PageUp, Hotkey::ScrollUp),
    (Code::PageDown, Hotkey::ScrollDown),
    (Code::Digit1, Hotkey::Monitor(0)),
    (Code::Digit2, Hotkey::Monitor(1)),
    (Code::Digit3, Hotkey::Monitor(2)),
    (Code::Digit4, Hotkey::Monitor(3)),
];

/// Key combinations shown in the help card; the descriptions live in `strings.rs`, in the same order.
pub const HOTKEY_KEYS: [&str; 8] = [
    "Ctrl+Shift+Space",
    "Ctrl+Shift+Enter",
    "Ctrl+Shift+H",
    "Ctrl+Shift+T",
    "Ctrl+Shift+L",
    "Ctrl+Shift+N",
    "Ctrl+Shift+PgUp/PgDn",
    "Ctrl+Shift+1..4",
];

fn is_wayland() -> bool {
    std::env::var("XDG_SESSION_TYPE").is_ok_and(|v| v.eq_ignore_ascii_case("wayland")) || std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty())
}

/// Registers the global hotkeys (X11). Wayland sessions get a log line instead.
pub fn spawn_hotkeys(tx: Sender<Event>, ctx: eframe::egui::Context) {
    let _ = CTX.set(ctx.clone());
    if is_wayland() {
        let _ = tx.send(Event::Log("global hotkeys are not available on Wayland: use the buttons in the window".into()));
        return;
    }
    std::thread::spawn(move || {
        let manager = match GlobalHotKeyManager::new() {
            Ok(m) => m,
            Err(e) => {
                let _ = tx.send(Event::Log(format!("could not start global hotkeys: {e}")));
                return;
            }
        };
        let mods = Some(Modifiers::CONTROL | Modifiers::SHIFT);
        let mut map: Vec<(u32, Hotkey)> = vec![];
        let mut failed = vec![];
        for &(code, hk) in KEYS {
            let key = HotKey::new(mods, code);
            if manager.register(key).is_ok() {
                map.push((key.id(), hk));
            } else {
                failed.push(format!("{code:?}"));
            }
        }
        if !failed.is_empty() {
            let _ = tx.send(Event::Log(format!("could not register hotkeys (taken by another application): {failed:?}")));
        }
        let tx2 = tx.clone();
        let ctx2 = ctx.clone();
        GlobalHotKeyEvent::set_event_handler(Some(move |e: GlobalHotKeyEvent| {
            if e.state != HotKeyState::Pressed {
                return;
            }
            let Some(&(_, hk)) = map.iter().find(|(id, _)| *id == e.id) else { return };
            match hk {
                // Handled right here: a hidden window may not get repaints.
                Hotkey::ToggleVisible => set_visible(!is_visible()),
                Hotkey::AskScreen | Hotkey::AskTranscript => {
                    set_visible(true);
                    let _ = tx2.send(Event::Hotkey(hk));
                }
                _ => {
                    let _ = tx2.send(Event::Hotkey(hk));
                }
            }
            ctx2.request_repaint();
        }));
        // The manager must stay alive for the registrations to hold.
        loop {
            std::thread::park();
            let _ = &manager;
        }
    });
}
