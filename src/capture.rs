use anyhow::Result;
use base64::Engine;
use image::{DynamicImage, ImageFormat, imageops::FilterType};
use std::io::Cursor;

#[derive(Clone, Debug)]
pub struct MonitorInfo {
    pub index: usize,
    pub width: u32,
    pub height: u32,
    pub x: i32,
    pub y: i32,
    pub primary: bool,
}

impl MonitorInfo {
    /// Label for lists; `primary` is the localized word for the main monitor.
    pub fn label(&self, primary: &str) -> String {
        let tail = if self.primary { format!(", {primary}") } else { String::new() };
        format!("{}: {}×{} @({},{}){}", self.index + 1, self.width, self.height, self.x, self.y, tail)
    }
}

#[cfg(not(windows))]
pub use crate::capture_linux::list_monitors;

#[cfg(windows)]
pub fn list_monitors() -> Vec<MonitorInfo> {
    let Ok(all) = xcap::Monitor::all() else { return vec![] };
    all.iter()
        .enumerate()
        .map(|(i, m)| {
            let w = m.width().unwrap_or(0);
            let h = m.height().unwrap_or(0);
            let x = m.x().unwrap_or(0);
            let y = m.y().unwrap_or(0);
            MonitorInfo { index: i, width: w, height: h, x, y, primary: m.is_primary().unwrap_or(false) }
        })
        .collect()
}

#[cfg(windows)]
fn grab(index: usize) -> Result<image::RgbImage> {
    let all = xcap::Monitor::all().map_err(|e| crate::errors::AppError::with(crate::errors::Kind::CaptureFailed, e))?;
    let m = all
        .get(index)
        .or_else(|| all.first())
        .ok_or_else(|| crate::errors::AppError::new(crate::errors::Kind::NoMonitors))?;
    let img = m.capture_image().map_err(|e| crate::errors::AppError::with(crate::errors::Kind::CaptureFailed, e))?;
    Ok(DynamicImage::ImageRgba8(img).to_rgb8())
}

#[cfg(not(windows))]
fn grab(index: usize) -> Result<image::RgbImage> {
    crate::capture_linux::grab(index).map_err(|e| match e.downcast::<crate::errors::AppError>() {
        Ok(app) => anyhow::Error::new(app),
        Err(e) => crate::errors::AppError::with(crate::errors::Kind::CaptureFailed, format!("{e:#}")),
    })
}

/// Captures a monitor and returns a base64 PNG (downscaled so the long edge <= max_edge).
pub fn capture_png_b64(index: usize, max_edge: u32) -> Result<String> {
    let mut img = grab(index)?;
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
