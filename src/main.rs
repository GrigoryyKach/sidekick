#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod agent;
mod app;
mod audio;
mod capture;
mod claude;
mod config;
mod errors;
mod events;
mod i18n;
mod nimvoro;
mod stt;
mod strings;
// Shared with the other NIMVORO tools: some of its widgets are not used here.
#[allow(dead_code)]
mod theme;
mod win;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let mut cfg = config::Config::load();
    // Contract: config language, then the launcher marker, then English; remember the result.
    let lang = i18n::resolve(cfg.language.as_deref());
    if cfg.language.as_deref() != Some(lang.code()) {
        cfg.language = Some(lang.code().to_string());
        cfg.save();
    }
    // Unique (invisible) title so we can find our HWND and apply overlay styles.
    let title = format!("Sidekick-{}", std::process::id());

    let mut vp = egui::ViewportBuilder::default()
        .with_title(&title)
        .with_app_id("local.sidekick")
        .with_inner_size(cfg.window_size)
        .with_min_inner_size([320.0, 240.0])
        .with_decorations(false)
        .with_transparent(true)
        .with_resizable(true)
        .with_taskbar(false)
        .with_window_level(egui::WindowLevel::AlwaysOnTop)
        .with_icon(std::sync::Arc::new(app_icon()));
    vp = match cfg.window_pos {
        Some(p) => vp.with_position(p),
        None => vp.with_position([80.0, 80.0]),
    };

    let options = eframe::NativeOptions { viewport: vp, ..Default::default() };
    eframe::run_native("Sidekick", options, Box::new(move |cc| Ok(Box::new(app::App::new(cc, cfg, title)))))
}

/// Window icon from `assets/icon.png` (the same artwork that is embedded in the exe).
fn app_icon() -> egui::IconData {
    eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon.png")).unwrap_or_default()
}
