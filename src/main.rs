#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod agent;
mod app;
mod audio;
mod capture;
mod claude;
mod config;
mod events;
mod nimvoro;
mod stt;
// Общий с остальными утилитами модуль оформления: часть компонентов здесь не используется.
#[allow(dead_code)]
mod theme;
mod win;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let cfg = config::Config::load();
    // Unique (invisible) title so we can find our HWND and apply overlay styles.
    let title = format!("transluent-{}", std::process::id());

    let mut vp = egui::ViewportBuilder::default()
        .with_title(&title)
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
    eframe::run_native(
        "transluent",
        options,
        Box::new(move |cc| Ok(Box::new(app::App::new(cc, cfg, title)))),
    )
}

fn app_icon() -> egui::IconData {
    let img = image::load_from_memory(include_bytes!("../assets/icon.png")).expect("assets/icon.png").to_rgba8();
    let (width, height) = img.dimensions();
    egui::IconData { rgba: img.into_raw(), width, height }
}
