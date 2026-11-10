//! Win32 helpers: capture exclusion, tool-window style, show/hide, global hotkeys.

use crate::events::{Event, Hotkey};
use crossbeam_channel::Sender;
use std::sync::atomic::{AtomicIsize, Ordering};
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    MOD_CONTROL, MOD_NOREPEAT, MOD_SHIFT, RegisterHotKey,
};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::PCWSTR;

static HWND_VAL: AtomicIsize = AtomicIsize::new(0);

fn hwnd() -> Option<HWND> {
    let v = HWND_VAL.load(Ordering::Relaxed);
    (v != 0).then(|| HWND(v as *mut _))
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Finds our window by its (invisible) unique title and applies the overlay styles.
/// Returns true once the window was found.
pub fn init_window(title: &str) -> bool {
    let t = wide(title);
    let Ok(h) = (unsafe { FindWindowW(PCWSTR::null(), PCWSTR(t.as_ptr())) }) else {
        return false;
    };
    if h.0.is_null() {
        return false;
    }
    HWND_VAL.store(h.0 as isize, Ordering::Relaxed);
    unsafe {
        // Excluded from screen capture / screen sharing (Windows 10 2004+).
        if std::env::var_os("TRANSLUENT_SHOW_IN_CAPTURE").is_none() {
            let _ = SetWindowDisplayAffinity(h, WDA_EXCLUDEFROMCAPTURE);
        }
        // Tool window: no taskbar button, not in Alt+Tab.
        let ex = GetWindowLongPtrW(h, GWL_EXSTYLE);
        let ex = (ex | WS_EX_TOOLWINDOW.0 as isize) & !(WS_EX_APPWINDOW.0 as isize);
        SetWindowLongPtrW(h, GWL_EXSTYLE, ex);
        let _ = SetWindowPos(
            h,
            Some(HWND_TOPMOST),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        );
    }
    true
}

/// winit re-applies its own window styles on some events; put ours back if they got lost.
pub fn ensure_tool_window() {
    let Some(h) = hwnd() else { return };
    unsafe {
        let ex = GetWindowLongPtrW(h, GWL_EXSTYLE);
        if ex & WS_EX_TOOLWINDOW.0 as isize == 0 {
            let ex = (ex | WS_EX_TOOLWINDOW.0 as isize) & !(WS_EX_APPWINDOW.0 as isize);
            SetWindowLongPtrW(h, GWL_EXSTYLE, ex);
            let _ = SetWindowPos(h, Some(HWND_TOPMOST), 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED);
        }
    }
}

pub fn is_capture_excluded() -> bool {
    let Some(h) = hwnd() else { return false };
    let mut aff = 0u32;
    unsafe { GetWindowDisplayAffinity(h, &mut aff).is_ok() && aff == WDA_EXCLUDEFROMCAPTURE.0 }
}

pub fn is_visible() -> bool {
    hwnd().map(|h| unsafe { IsWindowVisible(h).as_bool() }).unwrap_or(true)
}

pub fn set_visible(v: bool) {
    if let Some(h) = hwnd() {
        unsafe {
            let _ = ShowWindow(h, if v { SW_SHOWNOACTIVATE } else { SW_HIDE });
        }
    }
}

const KEYS: &[(i32, u32, Hotkey)] = &[
    (1, 0x20, Hotkey::AskScreen),     // Space
    (2, 0x0D, Hotkey::AskTranscript), // Enter
    (3, 0x48, Hotkey::ToggleVisible), // H
    (4, 0x54, Hotkey::ToggleClickThrough), // T
    (5, 0x4C, Hotkey::ToggleListen),  // L
    (6, 0x4E, Hotkey::NewChat),       // N
    (7, 0x21, Hotkey::ScrollUp),      // PgUp
    (8, 0x22, Hotkey::ScrollDown),    // PgDn
    (11, 0x31, Hotkey::Monitor(0)),   // 1
    (12, 0x32, Hotkey::Monitor(1)),   // 2
    (13, 0x33, Hotkey::Monitor(2)),   // 3
    (14, 0x34, Hotkey::Monitor(3)),   // 4
];

pub const HOTKEY_HELP: &[(&str, &str)] = &[
    ("Ctrl+Shift+Space", "спросить: экран + речь"),
    ("Ctrl+Shift+Enter", "спросить: только речь"),
    ("Ctrl+Shift+H", "скрыть / показать окно"),
    ("Ctrl+Shift+T", "сквозные клики вкл/выкл"),
    ("Ctrl+Shift+L", "прослушивание вкл/выкл"),
    ("Ctrl+Shift+N", "новый диалог"),
    ("Ctrl+Shift+PgUp/PgDn", "прокрутка ответа"),
    ("Ctrl+Shift+1..4", "выбрать монитор"),
];

/// Registers global hotkeys on a dedicated thread with its own message loop.
pub fn spawn_hotkeys(tx: Sender<Event>, ctx: eframe::egui::Context) {
    std::thread::spawn(move || unsafe {
        let mut failed = vec![];
        for &(id, vk, _) in KEYS {
            if RegisterHotKey(None, id, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, vk).is_err() {
                failed.push(id);
            }
        }
        if !failed.is_empty() {
            let _ = tx.send(Event::Log(format!(
                "не удалось зарегистрировать горячие клавиши (заняты другим приложением): {failed:?}"
            )));
        }
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            if msg.message == WM_HOTKEY {
                let id = msg.wParam.0 as i32;
                if let Some(&(_, _, hk)) = KEYS.iter().find(|k| k.0 == id) {
                    match hk {
                        // Handled right here: a hidden window may not get repaints.
                        Hotkey::ToggleVisible => set_visible(!is_visible()),
                        Hotkey::AskScreen | Hotkey::AskTranscript => {
                            set_visible(true);
                            let _ = tx.send(Event::Hotkey(hk));
                        }
                        _ => {
                            let _ = tx.send(Event::Hotkey(hk));
                        }
                    }
                    ctx.request_repaint();
                }
            }
        }
    });
}
