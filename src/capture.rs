use anyhow::{Context, Result, anyhow};
use base64::Engine;
use image::{DynamicImage, ImageFormat, imageops::FilterType};
use std::io::Cursor;

#[derive(Clone, Debug)]
pub struct MonitorInfo {
    pub label: String,
}

pub fn list_monitors() -> Vec<MonitorInfo> {
    let Ok(all) = xcap::Monitor::all() else { return vec![] };
    all.iter()
        .enumerate()
        .map(|(i, m)| {
            let w = m.width().unwrap_or(0);
            let h = m.height().unwrap_or(0);
            let x = m.x().unwrap_or(0);
            let y = m.y().unwrap_or(0);
            let primary = if m.is_primary().unwrap_or(false) { ", основной" } else { "" };
            MonitorInfo {
                label: format!("{}: {}×{} @({x},{y}){primary}", i + 1, w, h),
            }
        })
        .collect()
}

/// Captures a monitor and returns a base64 PNG (downscaled so the long edge <= max_edge).
pub fn capture_png_b64(index: usize, max_edge: u32) -> Result<String> {
    let all = xcap::Monitor::all().map_err(|e| anyhow!("{e}"))?;
    let m = all
        .get(index)
        .or_else(|| all.first())
        .context("мониторы не найдены")?;
    let img = m.capture_image().map_err(|e| anyhow!("захват экрана: {e}"))?;
    let mut img = DynamicImage::ImageRgba8(img).to_rgb8();
    let (w, h) = img.dimensions();
    let long = w.max(h);
    if long > max_edge {
        let k = max_edge as f32 / long as f32;
        img = image::imageops::resize(
            &img,
            (w as f32 * k) as u32,
            (h as f32 * k) as u32,
            FilterType::Triangle,
        );
    }
    let mut buf = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(img).write_to(&mut buf, ImageFormat::Png)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(buf.into_inner()))
}
