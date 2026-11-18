//! Linux screen capture. X11: the root window is read through the X protocol (`x11rb`, pure Rust, per RandR monitor).
//! Wayland: programs cannot read the screen, so the first of the usual screenshot tools is used
//! (`grim` on wlroots compositors, `gnome-screenshot` on GNOME, `spectacle` on KDE).
//!
//! `xcap` is used on Windows only: its Linux part needs libpipewire 1.x, newer than Ubuntu 22.04 has.

use crate::capture::MonitorInfo;
use anyhow::{Context, Result, anyhow, bail};
use image::RgbImage;
use x11rb::connection::Connection;
use x11rb::protocol::randr::ConnectionExt as _;
use x11rb::protocol::xproto::{ConnectionExt as _, ImageFormat};

fn is_wayland() -> bool {
    std::env::var("XDG_SESSION_TYPE").is_ok_and(|v| v.eq_ignore_ascii_case("wayland")) || std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty())
}

/// Monitors from RandR 1.5 (through XWayland on Wayland sessions); the whole screen when RandR is too old.
pub fn list_monitors() -> Vec<MonitorInfo> {
    x11_monitors().unwrap_or_else(|| if is_wayland() { vec![MonitorInfo { index: 0, width: 0, height: 0, x: 0, y: 0, primary: true }] } else { vec![] })
}

fn x11_monitors() -> Option<Vec<MonitorInfo>> {
    let (conn, num) = x11rb::connect(None).ok()?;
    let screen = &conn.setup().roots[num];
    let list = conn.randr_get_monitors(screen.root, true).ok().and_then(|c| c.reply().ok());
    let mut out: Vec<MonitorInfo> = match list {
        Some(r) => r
            .monitors
            .iter()
            .enumerate()
            .map(|(i, m)| MonitorInfo { index: i, width: m.width as u32, height: m.height as u32, x: m.x as i32, y: m.y as i32, primary: m.primary })
            .collect(),
        None => vec![],
    };
    if out.is_empty() {
        out.push(MonitorInfo { index: 0, width: screen.width_in_pixels as u32, height: screen.height_in_pixels as u32, x: 0, y: 0, primary: true });
    }
    Some(out)
}

/// Captures monitor `index` (the first one when it does not exist).
pub fn grab(index: usize) -> Result<RgbImage> {
    if is_wayland() {
        return grab_wayland();
    }
    grab_x11(index)
}

fn grab_x11(index: usize) -> Result<RgbImage> {
    let monitors = x11_monitors().ok_or_else(|| crate::errors::AppError::new(crate::errors::Kind::NoMonitors))?;
    let m = monitors.get(index).or_else(|| monitors.first()).ok_or_else(|| crate::errors::AppError::new(crate::errors::Kind::NoMonitors))?;
    let (conn, num) = x11rb::connect(None).context("cannot connect to the X server")?;
    let setup = conn.setup();
    let screen = &setup.roots[num];
    let bpp = setup.pixmap_formats.iter().find(|f| f.depth == screen.root_depth).map(|f| f.bits_per_pixel).unwrap_or(32);
    if bpp != 32 && bpp != 24 {
        bail!("unsupported pixel format: {bpp} bits per pixel");
    }
    let (w, h) = (m.width as usize, m.height as usize);
    let mut pixels = Vec::with_capacity(w * h * 3);
    // Row bands keep every reply well below the maximum request size of the server.
    let band = (4_000_000 / (w.max(1) * 4)).clamp(1, h.max(1));
    let mut y = 0;
    while y < h {
        let rows = band.min(h - y);
        let reply = conn
            .get_image(ImageFormat::Z_PIXMAP, screen.root, m.x as i16, (m.y + y as i32) as i16, w as u16, rows as u16, !0)?
            .reply()
            .context("GetImage failed")?;
        pixels.extend(bgrx_to_rgb(&reply.data, (bpp / 8) as usize));
        y += rows;
    }
    RgbImage::from_raw(w as u32, h as u32, pixels).ok_or_else(|| anyhow!("unexpected image size"))
}

/// X server pixels are `B G R (X)` on little-endian machines.
fn bgrx_to_rgb(data: &[u8], bytes_per_pixel: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / bytes_per_pixel * 3);
    for px in data.chunks_exact(bytes_per_pixel) {
        out.extend_from_slice(&[px[2], px[1], px[0]]);
    }
    out
}

fn grab_wayland() -> Result<RgbImage> {
    let file = std::env::temp_dir().join(format!("sidekick-shot-{}.png", std::process::id()));
    let path = file.display().to_string();
    let tools: [(&str, Vec<String>); 3] = [
        ("grim", vec![path.clone()]),
        ("gnome-screenshot", vec!["-f".into(), path.clone()]),
        ("spectacle", vec!["-b".into(), "-n".into(), "-f".into(), "-o".into(), path.clone()]),
    ];
    let mut tried = vec![];
    for (tool, args) in tools {
        let ran = std::process::Command::new(tool)
            .args(&args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        match ran {
            Ok(s) if s.success() && file.exists() => {
                let img = image::open(&file).context("cannot read the screenshot")?.to_rgb8();
                let _ = std::fs::remove_file(&file);
                return Ok(img);
            }
            _ => tried.push(tool),
        }
    }
    bail!("no screenshot tool worked on Wayland (tried {}): install grim, gnome-screenshot or spectacle", tried.join(", "))
}

#[cfg(test)]
mod tests {
    use super::bgrx_to_rgb;

    #[test]
    fn converts_bgrx_to_rgb() {
        assert_eq!(bgrx_to_rgb(&[1, 2, 3, 255, 4, 5, 6, 0], 4), vec![3, 2, 1, 6, 5, 4]);
        assert_eq!(bgrx_to_rgb(&[1, 2, 3, 4, 5, 6], 3), vec![3, 2, 1, 6, 5, 4]);
    }
}
