//! System-audio (WASAPI loopback) capture → mono 16 kHz → phrase segmentation by pauses.
//! Only what plays through the speakers/headphones is captured, i.e. the other call
//! participants — your own microphone never gets here.

use crate::events::Event;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded, unbounded};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub struct SegmenterCfg {
    pub threshold: f32,
    pub silence: Duration,
    pub max_len_s: f32,
}

pub struct AudioHandle {
    stop: Arc<AtomicBool>,
}

impl Drop for AudioHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

const SR: u32 = 16_000;

/// Starts loopback capture of the default output device. Segments (16 kHz mono f32)
/// are pushed to `seg_tx`.
pub fn start(
    cfg: SegmenterCfg,
    seg_tx: Sender<Vec<f32>>,
    ev: Sender<Event>,
    ctx: eframe::egui::Context,
) -> AudioHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    std::thread::spawn(move || {
        if let Err(e) = run(cfg, seg_tx, ev.clone(), ctx.clone(), stop2) {
            let _ = ev.send(Event::AudioError(e.to_string()));
            ctx.request_repaint();
        }
    });
    AudioHandle { stop }
}

fn run(
    cfg: SegmenterCfg,
    seg_tx: Sender<Vec<f32>>,
    ev: Sender<Event>,
    ctx: eframe::egui::Context,
    stop: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let dev = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no audio output device"))?;
    let name = dev.description().map(|d| d.to_string()).unwrap_or_default();
    let sup = dev.default_output_config()?;
    let channels = sup.channels() as usize;
    let in_sr = sup.sample_rate();
    let fmt = sup.sample_format();
    let config = sup.config();

    let (raw_tx, raw_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();
    let err_ev = ev.clone();
    let err_cb = move |e: cpal::Error| {
        let _ = err_ev.send(Event::Log(format!("audio (warning): {e}")));
    };
    let stream = match fmt {
        cpal::SampleFormat::F32 => dev.build_input_stream::<f32, _, _>(
            config,
            move |d: &[f32], _| {
                let _ = raw_tx.send(d.to_vec());
            },
            err_cb,
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_input_stream::<i16, _, _>(
            config,
            move |d: &[i16], _| {
                let _ = raw_tx.send(d.iter().map(|&s| s as f32 / 32768.0).collect());
            },
            err_cb,
            None,
        )?,
        other => anyhow::bail!("unsupported audio format: {other:?}"),
    };
    stream.play()?;
    let _ = ev.send(Event::Log(format!(
        "audio: loopback \"{name}\", {in_sr} Hz, {channels} ch."
    )));

    let mut rs = Resampler::new(in_sr, SR);
    let mut seg = Segmenter::new(cfg);
    let mut last_level = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        match raw_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(buf) => {
                let mono: Vec<f32> = buf
                    .chunks(channels)
                    .map(|c| c.iter().sum::<f32>() / channels as f32)
                    .collect();
                let s16 = rs.process(&mono);
                for out in seg.push(&s16) {
                    let _ = seg_tx.send(out);
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                // WASAPI loopback delivers nothing while nothing is playing: treat as silence.
                if let Some(out) = seg.tick_silence() {
                    let _ = seg_tx.send(out);
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if last_level.elapsed() > Duration::from_millis(80) {
            last_level = Instant::now();
            let _ = ev.send(Event::AudioLevel(seg.level));
            ctx.request_repaint();
        }
    }
    drop(stream);
    Ok(())
}

struct Resampler {
    step: f64,
    pos: f64,
    last: f32,
}

impl Resampler {
    fn new(from: u32, to: u32) -> Self {
        Self { step: from as f64 / to as f64, pos: 0.0, last: 0.0 }
    }
    /// Linear interpolation with a crude box pre-filter (good enough for speech).
    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity((input.len() as f64 / self.step) as usize + 1);
        let get = |i: isize, last: f32| if i < 0 { last } else { input[i as usize] };
        let win = self.step.floor().max(1.0) as isize;
        while self.pos < input.len() as f64 {
            let i = self.pos.floor() as isize;
            let frac = (self.pos - i as f64) as f32;
            let a = get(i - 1, self.last);
            let b = get(i, self.last);
            let mut acc = a + (b - a) * frac;
            if win > 1 {
                let mut s = 0.0;
                for k in 0..win {
                    s += get(i - k, self.last);
                }
                acc = 0.5 * acc + 0.5 * s / win as f32;
            }
            out.push(acc);
            self.pos += self.step;
        }
        self.pos -= input.len() as f64;
        if let Some(&l) = input.last() {
            self.last = l;
        }
        out
    }
}

struct Segmenter {
    cfg: SegmenterCfg,
    pre: Vec<f32>,
    buf: Vec<f32>,
    frame: Vec<f32>,
    in_speech: bool,
    voiced_samples: usize,
    last_voice: Instant,
    level: f32,
}

const FRAME: usize = 480; // 30 ms
const PREROLL: usize = SR as usize / 3;

impl Segmenter {
    fn new(cfg: SegmenterCfg) -> Self {
        Self {
            cfg,
            pre: vec![],
            buf: vec![],
            frame: Vec::with_capacity(FRAME),
            in_speech: false,
            voiced_samples: 0,
            last_voice: Instant::now(),
            level: 0.0,
        }
    }

    fn push(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut outs = vec![];
        for &s in samples {
            self.frame.push(s);
            if self.frame.len() < FRAME {
                continue;
            }
            let rms = (self.frame.iter().map(|x| x * x).sum::<f32>() / FRAME as f32).sqrt();
            self.level = self.level * 0.7 + rms * 0.3;
            let voiced = rms > self.cfg.threshold;
            if voiced {
                self.last_voice = Instant::now();
                if !self.in_speech {
                    self.in_speech = true;
                    self.buf = std::mem::take(&mut self.pre);
                }
                self.voiced_samples += FRAME;
            }
            if self.in_speech {
                self.buf.extend_from_slice(&self.frame);
                let too_long = self.buf.len() as f32 / SR as f32 > self.cfg.max_len_s;
                if too_long || (!voiced && self.last_voice.elapsed() > self.cfg.silence) {
                    if let Some(o) = self.flush() {
                        outs.push(o);
                    }
                }
            } else {
                self.pre.extend_from_slice(&self.frame);
                if self.pre.len() > PREROLL {
                    let cut = self.pre.len() - PREROLL;
                    self.pre.drain(..cut);
                }
            }
            self.frame.clear();
        }
        outs
    }

    fn tick_silence(&mut self) -> Option<Vec<f32>> {
        self.level *= 0.5;
        if self.in_speech && self.last_voice.elapsed() > self.cfg.silence {
            return self.flush();
        }
        None
    }

    fn flush(&mut self) -> Option<Vec<f32>> {
        self.in_speech = false;
        let voiced = std::mem::take(&mut self.voiced_samples);
        let buf = std::mem::take(&mut self.buf);
        // Ignore clicks / very short blips (< 0.4 s of voiced audio).
        (voiced as f32 / SR as f32 >= 0.4).then_some(buf)
    }
}

/// Channel used between audio capture and the STT sidecar (bounded so a stuck
/// recogniser cannot eat all memory).
pub fn segment_channel() -> (Sender<Vec<f32>>, Receiver<Vec<f32>>) {
    bounded(64)
}
