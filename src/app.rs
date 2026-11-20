use crate::agent::{self, Agent};
use crate::audio::{self, AudioHandle, SegmenterCfg};
use crate::capture::{self, MonitorInfo};
use crate::config::{AgentKind, Config, DEFAULT_ANTHROPIC_MODEL, DEFAULT_ANTHROPIC_URL, DEFAULT_OPENAI_URL, default_stt_prompt, default_system_prompt};
use crate::errors::{self, AppError, Kind};
use crate::events::{Event, Hotkey};
use crate::i18n::{self, Lang};
use crate::nimvoro::{self, UpdateState, Updater};
use crate::stt::{self, Stt};
use crate::strings::{S, TABLE};
use crate::support;
use crate::update_ui;
use crate::theme::{self, Btn, Pal, ThemeMode};
use crate::win;
use crossbeam_channel::{Receiver, Sender, unbounded};
use eframe::egui::{self, Align, Color32, Frame, Layout, Margin, RichText, Sense, Stroke, TextEdit, Ui, ViewportCommand, vec2};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cyan from the app icon. Light is the darker shade; both keep 4.5:1 as text on every surface.
const ACCENT: theme::Accent = theme::Accent::new(0x2cb5d6, 0x0e7490, 0x141415, 0xffffff);

struct Answer {
    title: String,
    with_screen: bool,
    text: String,
    done: bool,
    error: bool,
}

struct Line {
    at: Duration,
    text: String,
}

#[derive(Clone, Copy, PartialEq)]
enum AgentState {
    Off,
    Starting,
    Ready,
    Error,
    /// The process exited; it restarts on the next question.
    Stopped,
}

/// State of the speech recognizer. Texts are built at draw time so a language switch applies live.
#[derive(Clone, PartialEq)]
enum SttState {
    Off,
    Loading,
    Ready { model: String, device: String, load_s: String },
    Stopped,
    /// Start failed: the coded error (localized at draw time) or a raw message.
    Failed { app: Option<AppError>, raw: String },
    RestartApp,
}

#[derive(Clone, Copy, PartialEq)]
enum SettingsTab {
    Agent,
    Speech,
    Window,
    Prompts,
    About,
}

pub struct App {
    cfg: Config,
    lang: Lang,
    title: String,
    tx: Sender<Event>,
    rx: Receiver<Event>,
    started: Instant,
    window_ready: bool,
    capture_excluded: bool,
    updater: Updater,

    agent: Arc<Mutex<Option<Box<dyn Agent>>>>,
    agent_err: Option<anyhow::Error>,
    demo: bool,
    agent_state: AgentState,

    seg_tx: Sender<Vec<f32>>,
    seg_rx: Option<Receiver<Vec<f32>>>,
    stt: Option<Stt>,
    stt_state: SttState,
    audio: Option<AudioHandle>,
    level: f32,

    transcript: Vec<Line>,
    sent_upto: usize,
    answers: Vec<Answer>,
    md: CommonMarkCache,
    input: String,
    input_with_screen: bool,

    click_through: bool,
    show_settings: bool,
    settings_tab: SettingsTab,
    show_transcript: bool,
    show_key: bool,
    draft: Config,
    monitors: Vec<MonitorInfo>,
    logs: Vec<String>,
    scroll_to: Option<f32>,
    scroll_cur: f32,
    stick_bottom: bool,
    last_geom_save: Instant,
    last_style_check: Instant,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, cfg: Config, title: String) -> Self {
        let (tx, rx) = unbounded();
        let ctx = cc.egui_ctx.clone();
        theme::set_accent(ACCENT);
        theme::install(&ctx, cfg.theme);
        ctx.add_plugin(DefaultCursor);
        win::spawn_hotkeys(tx.clone(), ctx.clone());
        ctx.set_zoom_factor(cfg.font_size / 14.0);

        let updater = Updater::new("sidekick", "GrigoryyKach/sidekick", env!("CARGO_PKG_VERSION"));
        if cfg.check_updates {
            let c = ctx.clone();
            updater.check(move || c.request_repaint());
        }

        let (seg_tx, seg_rx) = audio::segment_channel();
        let mut app = Self {
            draft: cfg.clone(),
            lang: cfg.lang(),
            cfg,
            title,
            tx,
            rx,
            started: Instant::now(),
            window_ready: false,
            capture_excluded: false,
            updater,
            agent: Arc::new(Mutex::new(None)),
            agent_err: None,
            demo: std::env::var_os("SIDEKICK_DEMO").is_some(),
            agent_state: AgentState::Off,
            seg_tx,
            seg_rx: Some(seg_rx),
            stt: None,
            stt_state: SttState::Off,
            audio: None,
            level: 0.0,
            transcript: vec![],
            sent_upto: 0,
            answers: vec![],
            md: CommonMarkCache::default(),
            input: String::new(),
            input_with_screen: false,
            click_through: false,
            show_settings: false,
            settings_tab: SettingsTab::Agent,
            show_transcript: true,
            show_key: false,
            monitors: capture::list_monitors(),
            logs: vec![],
            scroll_to: None,
            scroll_cur: 0.0,
            stick_bottom: true,
            last_geom_save: Instant::now(),
            last_style_check: Instant::now(),
        };
        if app.demo {
            app.load_demo();
        }
        if app.cfg.listen_on_start {
            app.set_listening(true, &ctx);
        }
        app.ensure_agent(&ctx);
        app
    }

    /// Screenshot-only demo session (`SIDEKICK_DEMO=1`): a few transcript lines from "other participants"
    /// and a recognizer that looks loaded. Never used in normal runs.
    fn load_demo(&mut self) {
        let lines = [
            (12, "Let's start with the main question: how would you find two numbers in an array that add up to a target?"),
            (31, "The array is not sorted, and it can contain a few thousand elements."),
            (47, "What is the time complexity of your solution?"),
            (58, "And could we do it with less extra memory?"),
        ];
        for (secs, text) in lines {
            self.transcript.push(Line { at: Duration::from_secs(secs), text: text.to_string() });
        }
        self.stt_state = SttState::Ready { model: self.cfg.whisper_model.clone(), device: "CUDA".into(), load_s: "4.2".into() };
        self.answers.push(Answer {
            title: "Speech · 4 lines".into(),
            with_screen: false,
            text: "Use a hash map to find the pair in one pass. Time O(n), memory O(n).\n\n```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            return [seen[target - x], i]\n        seen[x] = i\n```\n\nSorting plus two pointers also works, in O(n log n) time and O(1) extra memory.".into(),
            done: true,
            error: false,
        });
        self.sent_upto = self.transcript.len();
    }

    /// Strings of the current interface language.
    fn s(&self) -> &'static S {
        TABLE.get(self.lang)
    }

    fn log(&mut self, s: impl Into<String>) {
        let s = s.into();
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(crate::config::app_dir().join("sidekick.log"))
        {
            use std::io::Write;
            let _ = writeln!(f, "[{:>7.1}s] {s}", self.started.elapsed().as_secs_f32());
        }
        self.logs.push(s);
        if self.logs.len() > 50 {
            self.logs.remove(0);
        }
    }

    // ---------- language and theme ----------

    fn set_language(&mut self, lang: Lang) {
        self.lang = lang;
        self.cfg.language = Some(lang.code().to_string());
        self.draft.language = self.cfg.language.clone();
        self.cfg.save();
    }

    fn set_theme(&mut self, ctx: &egui::Context, mode: ThemeMode) {
        theme::apply(ctx, mode);
        self.cfg.theme = mode;
        self.draft.theme = mode;
        self.cfg.save();
    }

    // ---------- agent ----------

    fn kind_title(&self, k: AgentKind) -> &'static str {
        self.s().agent_titles[k.index()]
    }

    fn agent_name(&self) -> String {
        let model = self.cfg.agent_model();
        let title = self.kind_title(self.cfg.agent);
        if model.is_empty() { title.to_string() } else { format!("{title} · {model}") }
    }

    fn agent_status_text(&self) -> String {
        let s = self.s();
        match self.agent_state {
            AgentState::Off => s.agent_none.to_string(),
            AgentState::Starting => format!("{}: {}", self.agent_name(), s.agent_starting),
            AgentState::Ready => format!("{}: {}", self.agent_name(), s.agent_ready),
            AgentState::Stopped => format!("{}: {}", self.kind_title(self.cfg.agent), s.agent_stopped),
            AgentState::Error => {
                let detail = self.agent_err.as_ref().map(|e| errors::localize(e, self.lang)).unwrap_or_default();
                format!("{}: {}", self.kind_title(self.cfg.agent), detail)
            }
        }
    }

    fn stt_status_text(&self) -> String {
        let s = self.s();
        match &self.stt_state {
            SttState::Off => s.whisper_off.to_string(),
            SttState::Loading => s.whisper_loading.replace("{model}", &self.cfg.whisper_model),
            SttState::Ready { model, device, load_s } => {
                s.whisper_ready.replace("{model}", model).replace("{device}", device).replace("{s}", load_s)
            }
            SttState::Stopped => format!("Whisper: {}", s.err_stt_exited),
            SttState::Failed { app, raw } => match app {
                Some(a) => format!("Whisper: {}", a.text(self.lang)),
                None => format!("Whisper: {raw}"),
            },
            SttState::RestartApp => s.whisper_restart.to_string(),
        }
    }

    fn ensure_agent(&mut self, ctx: &egui::Context) -> bool {
        let mut g = self.agent.lock();
        if let Some(a) = g.as_mut() {
            if a.alive() {
                return true;
            }
        }
        match agent::start(&self.cfg, self.tx.clone(), ctx.clone()) {
            Ok(a) => {
                *g = Some(a);
                self.agent_state = AgentState::Starting;
                true
            }
            Err(e) => {
                *g = None;
                self.agent_err = Some(e);
                self.agent_state = AgentState::Error;
                false
            }
        }
    }

    fn restart_agent(&mut self, ctx: &egui::Context) {
        *self.agent.lock() = None;
        for a in self.answers.iter_mut().filter(|a| !a.done) {
            a.done = true;
        }
        self.ensure_agent(ctx);
    }

    fn new_chat(&mut self, ctx: &egui::Context) {
        self.answers.clear();
        self.sent_upto = self.transcript.len();
        self.restart_agent(ctx);
    }

    fn ask(&mut self, with_screen: bool, user_text: Option<String>, ctx: &egui::Context) {
        let s = self.s();
        if !self.ensure_agent(ctx) {
            self.answers.push(Answer {
                title: s.no_agent_title.into(),
                with_screen: false,
                text: format!("{}\n\n{}", self.agent_status_text(), s.no_agent_hint),
                done: true,
                error: true,
            });
            return;
        }
        let new_lines: Vec<String> = self.transcript[self.sent_upto..].iter().map(|l| format!("- {}", l.text)).collect();
        self.sent_upto = self.transcript.len();

        // What the agent receives is not interface copy and stays in English.
        let mut text = String::new();
        if !new_lines.is_empty() {
            text += "New lines from the other participants (transcript):\n";
            text += &new_lines.join("\n");
            text += "\n\n";
        }
        match &user_text {
            Some(q) => text += &format!("My question: {q}"),
            None if with_screen => text += "Look at the screen (screenshot) and help: solve the task or answer the other participant's question.",
            None if new_lines.is_empty() => text += "There are no new lines. Continue or clarify your previous answer.",
            None => text += "Help me answer the other participant.",
        }

        let lines = i18n::plural(self.lang, new_lines.len(), &s.lines);
        let title = match (&user_text, with_screen) {
            (Some(q), _) => truncate(q, 60),
            (None, true) => s.title_screen.replace("{m}", &(self.cfg.monitor_index + 1).to_string()).replace("{x}", &lines),
            (None, false) => s.title_speech.replace("{x}", &lines),
        };
        self.answers.push(Answer { title, with_screen, text: String::new(), done: false, error: false });
        self.stick_bottom = true;

        let agent = self.agent.clone();
        let (tx, ctx2) = (self.tx.clone(), ctx.clone());
        let (mon, edge) = (self.cfg.monitor_index, self.cfg.max_image_edge);
        let lang = self.lang;
        std::thread::spawn(move || {
            let img = if with_screen {
                match capture::capture_png_b64(mon, edge) {
                    Ok(i) => Some(i),
                    Err(e) => {
                        let _ = tx.send(Event::Log(format!("screenshot: {e}")));
                        None
                    }
                }
            } else {
                None
            };
            let res = match agent.lock().as_mut() {
                Some(a) => a.send(&text, img.as_deref()),
                None => Err(AppError::new(Kind::AgentNotRunning)),
            };
            if let Err(e) = res {
                let _ = tx.send(Event::AgentDone { text: s.send_failed.replace("{e}", &errors::localize(&e, lang)), is_error: true });
            }
            ctx2.request_repaint();
        });
    }

    // ---------- audio / recognition ----------

    fn set_listening(&mut self, on: bool, ctx: &egui::Context) {
        if !on {
            self.audio = None;
            self.level = 0.0;
            return;
        }
        if self.stt.is_none() {
            let Some(rx) = self.seg_rx.take() else {
                self.stt_state = SttState::RestartApp;
                return;
            };
            match stt::start(
                &self.cfg.python_path,
                &self.cfg.whisper_model,
                &self.cfg.stt_language,
                &self.cfg.whisper_device,
                &self.cfg.stt_prompt,
                rx,
                self.tx.clone(),
                ctx.clone(),
            ) {
                Ok(s) => {
                    self.stt = Some(s);
                    self.stt_state = SttState::Loading;
                }
                Err(e) => {
                    let app = e.downcast_ref::<AppError>().cloned();
                    self.stt_state = SttState::Failed { app, raw: e.to_string() };
                    return;
                }
            }
        }
        self.audio = Some(audio::start(
            SegmenterCfg {
                threshold: self.cfg.vad_threshold,
                silence: Duration::from_millis(self.cfg.silence_ms),
                max_len_s: self.cfg.max_segment_s,
            },
            self.seg_tx.clone(),
            self.tx.clone(),
            ctx.clone(),
        ));
    }

    fn restart_stt(&mut self, ctx: &egui::Context) {
        let listening = self.audio.is_some();
        self.audio = None;
        self.stt = None;
        let (seg_tx, seg_rx) = audio::segment_channel();
        self.seg_tx = seg_tx;
        self.seg_rx = Some(seg_rx);
        self.stt_state = SttState::Off;
        if listening {
            self.set_listening(true, ctx);
        }
    }

    // ---------- events ----------

    fn handle_events(&mut self, ctx: &egui::Context) {
        while let Ok(e) = self.rx.try_recv() {
            match e {
                Event::Hotkey(h) => self.on_hotkey(h, ctx),
                Event::Transcript(t) => {
                    self.transcript.push(Line { at: self.started.elapsed(), text: t });
                    if self.transcript.len() > 500 {
                        self.transcript.remove(0);
                        self.sent_upto = self.sent_upto.saturating_sub(1);
                    }
                }
                Event::SttReady { model, device, load_s } => {
                    self.log(format!("Whisper {model} on {device} (loaded in {load_s} s)"));
                    self.stt_state = SttState::Ready { model, device, load_s };
                }
                Event::SttStopped => {
                    self.stt_state = SttState::Stopped;
                    self.stt = None;
                    self.audio = None;
                    let (seg_tx, seg_rx) = audio::segment_channel();
                    self.seg_tx = seg_tx;
                    self.seg_rx = Some(seg_rx);
                }
                Event::AudioLevel(l) => self.level = l,
                Event::AudioError(s) => {
                    self.log(format!("audio: {s}"));
                    self.audio = None;
                }
                Event::AgentReady => {
                    self.agent_state = AgentState::Ready;
                }
                Event::AgentDelta(t) => {
                    if let Some(a) = self.answers.iter_mut().find(|a| !a.done) {
                        a.text += &t;
                    }
                }
                Event::AgentFailed(e) => {
                    let text = errors::localize(&e, self.lang);
                    self.log(format!("agent error: {e}"));
                    if let Some(a) = self.answers.iter_mut().find(|a| !a.done) {
                        a.done = true;
                        a.error = true;
                        if !a.text.is_empty() {
                            a.text += "

";
                        }
                        a.text += &text;
                    }
                }
                Event::AgentDone { text, is_error } => {
                    if let Some(a) = self.answers.iter_mut().find(|a| !a.done) {
                        a.done = true;
                        a.error = is_error;
                        if a.text.trim().is_empty() || is_error {
                            if !a.text.is_empty() {
                                a.text += "\n\n";
                            }
                            a.text += &text;
                        }
                    }
                }
                Event::AgentExited(tail) => {
                    let exited = self.s().agent_exited;
                    let was_running = self.agent.lock().as_mut().map(|c| !c.alive()).unwrap_or(false);
                    for a in self.answers.iter_mut().filter(|a| !a.done) {
                        a.done = true;
                        a.error = true;
                        a.text += &format!("\n\n{exited}\n```\n{tail}\n```");
                    }
                    if was_running {
                        *self.agent.lock() = None;
                        self.agent_state = AgentState::Stopped;
                        if !tail.trim().is_empty() {
                            self.log(format!("agent: {tail}"));
                        }
                    }
                }
                Event::Log(s) => self.log(s),
            }
        }
    }

    fn on_hotkey(&mut self, h: Hotkey, ctx: &egui::Context) {
        match h {
            Hotkey::AskScreen => self.ask(true, None, ctx),
            Hotkey::AskTranscript => self.ask(false, None, ctx),
            Hotkey::ToggleVisible => {}
            Hotkey::ToggleClickThrough => {
                self.click_through = !self.click_through;
                ctx.send_viewport_cmd(ViewportCommand::MousePassthrough(self.click_through));
            }
            Hotkey::ToggleListen => {
                let on = self.audio.is_none();
                self.set_listening(on, ctx);
            }
            Hotkey::NewChat => self.new_chat(ctx),
            Hotkey::ScrollUp => self.scroll_to = Some((self.scroll_cur - 300.0).max(0.0)),
            Hotkey::ScrollDown => self.scroll_to = Some(self.scroll_cur + 300.0),
            Hotkey::Monitor(i) => {
                if i < self.monitors.len().max(1) {
                    self.cfg.monitor_index = i;
                    self.draft.monitor_index = i;
                    self.cfg.save();
                }
            }
        }
    }

    // ---------- interface ----------

    /// Semi-transparent card fill: follows the window opacity setting.
    fn card_fill(&self, p: &Pal) -> Color32 {
        let a = (self.cfg.opacity * 0.85 * 255.0).clamp(30.0, 235.0) as u8;
        let [r, g, b, _] = p.card.to_array();
        Color32::from_rgba_unmultiplied(r, g, b, a)
    }

    /// A card on the translucent window: follows the opacity setting, hairline outline, radius 12.
    fn card_frame(&self, p: &Pal, margin: Margin) -> Frame {
        Frame::new().fill(self.card_fill(p)).stroke(Stroke::new(1.0, p.line)).corner_radius(theme::R_CARD).inner_margin(margin)
    }

    fn header(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            theme::mark(ui, 20.0, icon_headset, p);
            let r = ui.add(
                egui::Label::new(RichText::new("Sidekick").font(theme::medium(14.0)).color(p.text))
                    .selectable(false)
                    .sense(Sense::click_and_drag()),
            );
            if r.drag_started() {
                ctx.send_viewport_cmd(ViewportCommand::StartDrag);
            }
            r.on_hover_text(s.drag_tip);
            // The empty part of the header drags the window too.
            let rest = ui.available_rect_before_wrap();
            let drag = ui.interact(rest.with_max_x(rest.max.x - 188.0), ui.scope_id().with("drag"), Sense::click_and_drag());
            if drag.drag_started() {
                ctx.send_viewport_cmd(ViewportCommand::StartDrag);
            }

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                if icon_button(ui, "close", icon_close, IconKind::Plain, 28.0, p).on_hover_text(s.exit).clicked() {
                    ctx.send_viewport_cmd(ViewportCommand::Close);
                }
                if icon_button(ui, "hide", icon_minus, IconKind::Plain, 28.0, p).on_hover_text(s.hide_tip).clicked() {
                    win::set_visible(false);
                }
                let kind = if self.show_settings { IconKind::Active } else { IconKind::Plain };
                if icon_button(ui, "gear", icon_gear, kind, 28.0, p).on_hover_text(s.settings).clicked() {
                    self.toggle_settings();
                }
                if theme::button(ui, s.new_chat, Btn::Small, p).on_hover_text("Ctrl+Shift+N").clicked() {
                    self.new_chat(&ctx);
                }
            });
        });
    }

    fn toggle_settings(&mut self) {
        self.show_settings = !self.show_settings;
        self.draft = self.cfg.clone();
        self.monitors = capture::list_monitors();
    }

    fn actions(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            if theme::button(ui, s.btn_screen, Btn::Primary, p).on_hover_text("Ctrl+Shift+Space").clicked() {
                self.ask(true, None, &ctx);
            }
            if theme::button(ui, s.btn_speech, Btn::Secondary, p).on_hover_text("Ctrl+Shift+Enter").clicked() {
                self.ask(false, None, &ctx);
            }
            egui::ComboBox::from_id_salt("mon")
                .selected_text(s.monitor_n.replace("{n}", &(self.cfg.monitor_index + 1).to_string()))
                .width(104.0)
                .show_ui(ui, |ui| {
                    for (i, m) in self.monitors.iter().enumerate() {
                        if ui.selectable_label(self.cfg.monitor_index == i, m.label(s.monitor_primary)).clicked() {
                            self.cfg.monitor_index = i;
                            self.draft.monitor_index = i;
                            self.cfg.save();
                        }
                    }
                })
                .response
                .on_hover_text(s.monitor_tip);

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let listening = self.audio.is_some() || self.demo;
                let mut on = listening;
                let r = theme::switch(ui, &mut on, p).on_hover_text(s.listen_tip);
                if r.changed() {
                    self.set_listening(on, &ctx);
                }
                ui.label(RichText::new(s.listen).font(theme::reg(13.0)).color(if listening { p.text } else { p.text2 }));
            });
        });
    }

    fn status_row(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(16.0, 4.0);
            if self.capture_excluded || self.demo {
                chip(ui, p, p.positive, false, s.chip_hidden);
            } else {
                let r = chip(ui, p, p.warning, false, s.chip_visible);
                if cfg!(not(windows)) {
                    // Linux has no way to exclude a window from screen capture.
                    r.on_hover_text(s.chip_linux_tip);
                }
            }
            let col = match self.agent_state {
                AgentState::Ready => p.positive,
                AgentState::Starting => p.warning,
                AgentState::Error => p.negative,
                AgentState::Off | AgentState::Stopped => p.muted,
            };
            chip(ui, p, col, false, &self.agent_status_text());
            // The signature: the recognition dot fades once every 1.6 s, and only while it is really listening.
            let capturing = self.audio.is_some() && self.stt.is_some();
            let stt_col = if capturing || self.demo { p.positive } else { p.muted };
            let r = chip(ui, p, stt_col, capturing, &self.stt_status_text());
            if self.audio.is_some() {
                // Audio level under the recognition status.
                let w = (self.level * 800.0).clamp(0.0, 1.0);
                let bar = egui::Rect::from_min_size(r.rect.left_bottom() + vec2(16.0, 1.0), vec2((r.rect.width() - 16.0).max(0.0), 2.0));
                ui.painter().rect_filled(bar, 1.0, p.line_hi);
                let mut fill = bar;
                fill.set_width(bar.width() * w);
                ui.painter().rect_filled(fill, 1.0, if self.level > self.cfg.vad_threshold { p.positive } else { p.muted });
            }
            if self.click_through {
                chip(ui, p, p.accent, false, s.chip_passthrough);
            }
        });
    }

    fn update_banner(&mut self, ui: &mut Ui, p: &Pal) {
        let UpdateState::Available { version, .. } = self.updater.state() else { return };
        let t = nimvoro::strings(self.lang);
        Frame::new()
            .fill(p.raised)
            .stroke(Stroke::new(1.0, p.line_hi))
            .corner_radius(theme::R_CARD)
            .inner_margin(Margin::symmetric(12, 8))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                if !update_ui::progress(ui, &self.updater, self.lang, p) {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(t.available.replace("{v}", &version)).font(theme::medium(13.0)).color(p.text));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if theme::button(ui, t.update_btn, Btn::Secondary, p).clicked() && self.updater.apply() {
                                self.cfg.save();
                                ui.ctx().send_viewport_cmd(ViewportCommand::Close);
                            }
                        });
                    });
                }
            });
    }

    fn transcript_view(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        let new = self.transcript.len() - self.sent_upto;
        self.card_frame(p, Margin::same(12)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let r = theme::disclosure(ui, s.transcript, self.show_transcript, p).on_hover_text(if self.show_transcript { s.collapse } else { s.expand });
                if r.clicked() {
                    self.show_transcript = !self.show_transcript;
                }
                if new > 0 {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        theme::meta(ui, &i18n::plural(self.lang, new, &s.new_lines), p);
                    });
                }
            });
            if !self.show_transcript {
                return;
            }
            egui::ScrollArea::vertical().id_salt("trs").max_height(112.0).stick_to_bottom(true).auto_shrink([false, true]).show(ui, |ui| {
                // The shared theme turns label selection off; the transcript must be copyable.
                ui.style_mut().interaction.selectable_labels = true;
                ui.spacing_mut().item_spacing = vec2(12.0, 4.0);
                if self.transcript.is_empty() {
                    theme::hint(ui, s.transcript_empty, p);
                }
                for (i, l) in self.transcript.iter().enumerate() {
                    let secs = l.at.as_secs();
                    let fresh = i >= self.sent_upto;
                    ui.horizontal_top(|ui| {
                        // A fixed mono column for the time stamps, so the lines start on one edge.
                        ui.allocate_ui_with_layout(vec2(36.0, 20.0), Layout::left_to_right(Align::Min), |ui| {
                            ui.label(RichText::new(format!("{:02}:{:02}", secs / 60, secs % 60)).font(theme::mono(12.0)).color(p.muted));
                        });
                        ui.add(egui::Label::new(RichText::new(&l.text).font(theme::reg(14.0)).color(if fresh { p.text } else { p.text2 })).wrap());
                    });
                }
            });
        });
    }

    fn answers_view(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        let avail = ui.available_height() - 52.0;
        let mut sa = egui::ScrollArea::vertical()
            .id_salt("ans")
            .max_height(avail.max(60.0))
            .auto_shrink([false, false])
            .stick_to_bottom(self.stick_bottom);
        if let Some(y) = self.scroll_to.take() {
            sa = sa.vertical_scroll_offset(y);
            self.stick_bottom = false;
        }
        let out = sa.show(ui, |ui| {
            // The shared theme turns label selection off; answers must be selectable and copyable.
            ui.style_mut().interaction.selectable_labels = true;
            ui.spacing_mut().item_spacing.y = 8.0;
            if self.answers.is_empty() {
                self.card_frame(p, Margin::same(16)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.spacing_mut().item_spacing.y = 4.0;
                    ui.label(RichText::new(s.ready_title).font(theme::medium(14.0)).color(p.text));
                    theme::hint(ui, s.ready_hint, p);
                    ui.add_space(8.0);
                    egui::Grid::new("keys").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                        for (k, d) in win::HOTKEY_KEYS.iter().zip(s.hotkeys.iter()) {
                            theme::keycap(ui, k, p);
                            ui.label(RichText::new(*d).font(theme::reg(12.0)).color(p.text2));
                            ui.end_row();
                        }
                    });
                });
            }
            for a in self.answers.iter() {
                let fill = self.card_fill(p);
                let stroke = if a.error { Stroke::new(1.0, p.negative) } else { Stroke::new(1.0, p.line) };
                Frame::new().fill(fill).stroke(stroke).corner_radius(theme::R_CARD).inner_margin(Margin::symmetric(16, 12)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    // Code blocks and inline code sit one step above the card, in the neutral ramp.
                    ui.visuals_mut().code_bg_color = p.raised;
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 8.0;
                        let col = if a.error { p.negative } else { p.text2 };
                        if a.with_screen {
                            let (rect, _) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
                            icon_camera(ui.painter(), rect.center(), col);
                        }
                        ui.add(egui::Label::new(RichText::new(&a.title).font(theme::medium(13.0)).color(col)).truncate());
                        if !a.done {
                            ui.add(egui::Spinner::new().size(12.0).color(p.accent));
                        }
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if a.done && theme::button(ui, s.copy, Btn::Small, p).clicked() {
                                ui.ctx().copy_text(a.text.clone());
                            }
                        });
                    });
                    ui.add_space(4.0);
                    if a.text.is_empty() && !a.done {
                        theme::hint(ui, s.thinking, p);
                    } else {
                        CommonMarkViewer::new().show(ui, &mut self.md, &a.text);
                    }
                });
            }
        });
        self.scroll_cur = out.state.clamped_offset().y;
        let max = (out.content_size.y - out.inner_rect.height()).max(0.0);
        if out.state.clamped_offset().y < max - 40.0 && ui.input(|i| i.smooth_scroll_delta.y > 0.0) {
            self.stick_bottom = false;
        } else if out.state.clamped_offset().y >= max - 5.0 {
            self.stick_bottom = true;
        }
    }

    fn input_bar(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            let kind = if self.input_with_screen { IconKind::Active } else { IconKind::Plain };
            if icon_button(ui, "attach", icon_camera, kind, 32.0, p).on_hover_text(s.attach_tip).clicked() {
                self.input_with_screen = !self.input_with_screen;
            }
            let w = ui.available_width() - 32.0 - ui.spacing().item_spacing.x;
            let r = ui.add(
                TextEdit::singleline(&mut self.input)
                    .hint_text(s.ask_hint)
                    .desired_width(w)
                    .margin(Margin::symmetric(12, 8))
                    .font(theme::reg(14.0)),
            );
            let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let has_text = !self.input.trim().is_empty();
            let send_kind = if has_text { IconKind::Filled } else { IconKind::Plain };
            let clicked = icon_button(ui, "send", icon_send, send_kind, 32.0, p).on_hover_text(s.send_tip).clicked();
            if (clicked || enter) && has_text {
                let q = std::mem::take(&mut self.input);
                self.ask(self.input_with_screen, Some(q), &ctx);
            }
        });
    }

    // ---------- settings ----------

    fn settings_window(&mut self, ctx: &egui::Context, p: &Pal) {
        let s = self.s();
        let mut close = false;
        let mut apply = false;
        // No native title bar (it would add a second, unstyled close button): the header is drawn here.
        egui::Window::new("sidekick-settings")
            .title_bar(false)
            .collapsible(false)
            .default_width(460.0)
            .min_width(420.0)
            .default_height(520.0)
            .frame(Frame::window(&ctx.style_of(ctx.theme())).fill(p.card).stroke(Stroke::new(1.0, p.line_hi)).corner_radius(theme::R_WINDOW).shadow(theme::shadow_for(p)).inner_margin(Margin::same(16)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(s.settings).font(theme::medium(14.0)).color(p.text));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if icon_button(ui, "sclose", icon_close, IconKind::Plain, 28.0, p).on_hover_text(s.close).clicked() {
                            close = true;
                        }
                    });
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    for (tab, name) in [
                        (SettingsTab::Agent, s.tab_agent),
                        (SettingsTab::Speech, s.tab_speech),
                        (SettingsTab::Window, s.tab_window),
                        (SettingsTab::Prompts, s.tab_prompts),
                        (SettingsTab::About, s.tab_about),
                    ] {
                        let kind = if self.settings_tab == tab { Btn::Secondary } else { Btn::Quiet };
                        if theme::button(ui, name, kind, p).clicked() {
                            self.settings_tab = tab;
                        }
                    }
                });
                ui.add_space(8.0);
                egui::ScrollArea::vertical().max_height(ui.available_height() - 52.0).auto_shrink([false, true]).show(ui, |ui| {
                    ui.set_width(ui.available_width() - 6.0);
                    match self.settings_tab {
                        SettingsTab::Agent => self.settings_agent(ui, p),
                        SettingsTab::Speech => self.settings_speech(ui, p),
                        SettingsTab::Window => self.settings_window_tab(ui, p),
                        SettingsTab::Prompts => self.settings_prompts(ui, p),
                        SettingsTab::About => self.settings_about(ui, p),
                    }
                });
                theme::divider(ui, p);
                ui.horizontal(|ui| {
                    let dirty = !same_config(&self.draft, &self.cfg);
                    let label = if dirty { s.apply } else { s.saved };
                    if theme::button(ui, label, if dirty { Btn::Primary } else { Btn::Quiet }, p).clicked() && dirty {
                        apply = true;
                    }
                    if dirty && theme::button(ui, s.revert, Btn::Link, p).clicked() {
                        self.draft = self.cfg.clone();
                    }
                });
            });
        if close {
            self.show_settings = false;
        }
        if apply {
            self.apply_settings(ctx);
        }
    }

    fn settings_agent(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        let d = &mut self.draft;
        theme::section(ui, s.who_answers, p);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
            for k in AgentKind::ALL {
                let kind = if d.agent == k { Btn::Secondary } else { Btn::Quiet };
                if theme::button(ui, s.agent_titles[k.index()], kind, p).clicked() {
                    d.agent = k;
                }
            }
        });
        ui.add_space(4.0);
        theme::hint(ui, s.agent_about[d.agent.index()], p);
        theme::divider(ui, p);

        match d.agent {
            AgentKind::ClaudeCode => {
                theme::section(ui, "Claude Code", p);
                field(ui, p, s.model, None, |ui| {
                    egui::ComboBox::from_id_salt("model").selected_text(&d.model).show_ui(ui, |ui| {
                        for m in ["sonnet", "opus", "haiku"] {
                            ui.selectable_value(&mut d.model, m.to_string(), m);
                        }
                    });
                });
                effort_field(ui, p, s, &mut d.effort);
                field(ui, p, s.claude_path, Some(s.claude_path_hint), |ui| {
                    ui.add(TextEdit::singleline(&mut d.claude_path).hint_text(s.auto).desired_width(f32::INFINITY));
                });
            }
            AgentKind::Anthropic => {
                theme::section(ui, "Anthropic API", p);
                field(ui, p, s.api_key, Some(s.api_key_hint_anthropic), |ui| {
                    key_edit(ui, p, s, &mut d.api_key, &mut self.show_key);
                });
                field(ui, p, s.model, None, |ui| {
                    let shown = if d.api_model.is_empty() { DEFAULT_ANTHROPIC_MODEL.to_string() } else { d.api_model.clone() };
                    egui::ComboBox::from_id_salt("amodel").selected_text(shown).show_ui(ui, |ui| {
                        for m in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2"] {
                            ui.selectable_value(&mut d.api_model, m.to_string(), m);
                        }
                    });
                });
                effort_field(ui, p, s, &mut d.effort);
                field(ui, p, s.api_url, Some(s.api_url_hint_anthropic), |ui| {
                    ui.add(TextEdit::singleline(&mut d.api_base_url).hint_text(DEFAULT_ANTHROPIC_URL).desired_width(f32::INFINITY));
                });
            }
            AgentKind::OpenAi => {
                theme::section(ui, s.server, p);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                    for (name, url) in [
                        ("OpenAI", DEFAULT_OPENAI_URL),
                        ("OpenRouter", "https://openrouter.ai/api/v1"),
                        ("Ollama", "http://localhost:11434/v1"),
                        ("LM Studio", "http://localhost:1234/v1"),
                    ] {
                        let kind = if d.api_base_url == url || (d.api_base_url.is_empty() && url == DEFAULT_OPENAI_URL) { Btn::Secondary } else { Btn::Small };
                        if theme::button(ui, name, kind, p).clicked() {
                            d.api_base_url = url.to_string();
                        }
                    }
                });
                field(ui, p, s.api_url, Some(s.api_url_hint_openai), |ui| {
                    ui.add(TextEdit::singleline(&mut d.api_base_url).hint_text(DEFAULT_OPENAI_URL).desired_width(f32::INFINITY));
                });
                field(ui, p, s.api_key, Some(s.api_key_hint_openai), |ui| {
                    key_edit(ui, p, s, &mut d.api_key, &mut self.show_key);
                });
                field(ui, p, s.model, Some(s.model_hint_openai), |ui| {
                    ui.add(TextEdit::singleline(&mut d.api_model).hint_text(s.model_name_hint).desired_width(f32::INFINITY));
                });
            }
            AgentKind::Command => {
                theme::section(ui, s.command, p);
                field(ui, p, s.cmdline, Some(s.cmdline_hint), |ui| {
                    ui.add(TextEdit::singleline(&mut d.command_line).hint_text("ollama run llama3.2").desired_width(f32::INFINITY));
                });
                theme::hint(ui, s.cmd_note, p);
            }
        }
        theme::divider(ui, p);
        theme::section(ui, s.answer, p);
        field(ui, p, s.answer_lang, Some(s.answer_lang_hint), |ui| {
            ui.add(TextEdit::singleline(&mut d.answer_language).hint_text(self.lang.native_name()).desired_width(f32::INFINITY));
        });
        field(ui, p, s.max_edge, None, |ui| {
            ui.add(egui::DragValue::new(&mut d.max_image_edge).range(800..=3000));
        });
    }

    fn settings_speech(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        let d = &mut self.draft;
        theme::section(ui, s.speech_rec, p);
        switch_row(ui, p, s.listen_start, s.listen_start_hint, &mut d.listen_on_start);
        field(ui, p, s.whisper_model, None, |ui| {
            egui::ComboBox::from_id_salt("wm").selected_text(&d.whisper_model).show_ui(ui, |ui| {
                for m in ["large-v3-turbo", "medium", "small", "large-v3"] {
                    ui.selectable_value(&mut d.whisper_model, m.to_string(), m);
                }
            });
        });
        field(ui, p, s.speech_lang, None, |ui| {
            egui::ComboBox::from_id_salt("sl").selected_text(&d.stt_language).show_ui(ui, |ui| {
                for m in ["auto", "en", "ru", "uk"] {
                    ui.selectable_value(&mut d.stt_language, m.to_string(), m);
                }
            });
        });
        field(ui, p, s.device, Some(s.device_hint), |ui| {
            egui::ComboBox::from_id_salt("dev").selected_text(&d.whisper_device).show_ui(ui, |ui| {
                for m in ["cuda", "cpu"] {
                    ui.selectable_value(&mut d.whisper_device, m.to_string(), m);
                }
            });
        });
        field(ui, p, s.python, Some(s.python_hint), |ui| {
            ui.add(TextEdit::singleline(&mut d.python_path).desired_width(f32::INFINITY));
        });
        theme::divider(ui, p);
        theme::section(ui, s.phrases, p);
        field(ui, p, s.vad, None, |ui| {
            ui.add(egui::Slider::new(&mut d.vad_threshold, 0.001..=0.05).logarithmic(true));
        });
        field(ui, p, s.silence, None, |ui| {
            ui.add(egui::DragValue::new(&mut d.silence_ms).range(300..=3000));
        });
        theme::divider(ui, p);
        theme::section(ui, s.whisper_prompt, p);
        theme::hint(ui, s.whisper_prompt_hint, p);
        ui.add(TextEdit::multiline(&mut d.stt_prompt).desired_rows(3).desired_width(f32::INFINITY));
        if theme::button(ui, s.reset_whisper_prompt, Btn::Link, p).clicked() {
            d.stt_prompt = default_stt_prompt(self.lang).into();
        }
    }

    fn settings_window_tab(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        let ctx = ui.ctx().clone();
        theme::section(ui, s.appearance, p);
        // Language and theme apply live and are saved at once; they are not part of the draft.
        let mut lang = self.lang;
        if i18n::language_row(ui, &mut lang, p) {
            self.set_language(lang);
        }
        let mut mode = self.cfg.theme;
        if i18n::theme_row(ui, &mut mode, self.lang, p) {
            self.set_theme(&ctx, mode);
        }
        theme::divider(ui, p);

        let s = self.s();
        let d = &mut self.draft;
        theme::section(ui, s.window, p);
        field(ui, p, s.shot_monitor, None, |ui| {
            egui::ComboBox::from_id_salt("smon")
                .selected_text(self.monitors.get(d.monitor_index).map(|m| m.label(s.monitor_primary)).unwrap_or("—".into()))
                .show_ui(ui, |ui| {
                    for (i, m) in self.monitors.iter().enumerate() {
                        ui.selectable_value(&mut d.monitor_index, i, m.label(s.monitor_primary));
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new(s.opacity).font(theme::reg(14.0)).color(p.text));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| theme::meta(ui, &format!("{}%", (d.opacity * 100.0).round()), p));
        });
        theme::slider(ui, &mut d.opacity, 0.15, 1.0, 0.05, p);
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new(s.text_size).font(theme::reg(14.0)).color(p.text));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| theme::meta(ui, &format!("{}", d.font_size.round()), p));
        });
        theme::slider(ui, &mut d.font_size, 10.0, 22.0, 1.0, p);
    }

    fn settings_prompts(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        let d = &mut self.draft;
        theme::section(ui, s.system_prompt, p);
        theme::hint(ui, s.system_prompt_hint, p);
        ui.add(TextEdit::multiline(&mut d.system_prompt).desired_rows(12).desired_width(f32::INFINITY).font(theme::reg(13.0)));
        if theme::button(ui, s.reset_prompt, Btn::Link, p).clicked() {
            d.system_prompt = default_system_prompt(self.lang).into();
        }
    }

    fn settings_about(&mut self, ui: &mut Ui, p: &Pal) {
        let s = self.s();
        let t = nimvoro::strings(self.lang);
        theme::section(ui, t.updates, p);
        switch_row(ui, p, t.check_on_start, t.check_on_start_hint, &mut self.draft.check_updates);
        ui.add_space(4.0);
        if !update_ui::progress(ui, &self.updater, self.lang, p) {
            ui.horizontal(|ui| {
                let mut status = self.updater.status_text(self.lang);
                if matches!(self.updater.state(), UpdateState::Available { .. }) {
                    status += &format!(" · {}", t.you_have.replace("{v}", self.updater.current));
                }
                theme::hint(ui, &status, p);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if let UpdateState::Available { .. } = self.updater.state() {
                        if theme::button(ui, t.update_btn, Btn::Secondary, p).clicked() {
                            let saved = self.cfg.save();
                            if self.updater.apply_if(saved) {
                                ui.ctx().send_viewport_cmd(ViewportCommand::Close);
                            }
                        }
                    } else if !self.updater.is_checking() && theme::button(ui, t.check_now, Btn::Link, p).clicked() {
                        let c = ui.ctx().clone();
                        self.updater.check(move || c.request_repaint());
                    }
                });
            });
        }
        theme::divider(ui, p);
        support::support_section(ui, self.lang);
        theme::divider(ui, p);
        theme::section(ui, s.files, p);
        theme::hint(ui, &s.files_hint.replace("{dir}", &crate::config::app_dir().display().to_string()), p);
        if theme::button(ui, s.open_folder, Btn::Secondary, p).clicked() {
            let opener = if cfg!(windows) { "explorer" } else { "xdg-open" };
            let _ = std::process::Command::new(opener).arg(crate::config::app_dir()).spawn();
        }
        if !self.logs.is_empty() {
            theme::divider(ui, p);
            theme::section(ui, s.log, p);
            for l in self.logs.iter().rev().take(30) {
                ui.label(RichText::new(l).font(theme::mono(12.0)).color(p.muted));
            }
        }
    }

    fn apply_settings(&mut self, ctx: &egui::Context) {
        let old = std::mem::replace(&mut self.cfg, self.draft.clone());
        ctx.set_zoom_factor(self.cfg.font_size / 14.0);
        let agent_changed = old.agent != self.cfg.agent
            || old.model != self.cfg.model
            || old.effort != self.cfg.effort
            || old.system_prompt != self.cfg.system_prompt
            || old.answer_language != self.cfg.answer_language
            || old.claude_path != self.cfg.claude_path
            || old.api_base_url != self.cfg.api_base_url
            || old.api_key != self.cfg.api_key
            || old.api_model != self.cfg.api_model
            || old.command_line != self.cfg.command_line;
        let stt_changed = old.whisper_model != self.cfg.whisper_model
            || old.stt_language != self.cfg.stt_language
            || old.whisper_device != self.cfg.whisper_device
            || old.python_path != self.cfg.python_path
            || old.stt_prompt != self.cfg.stt_prompt;
        let audio_changed = old.vad_threshold != self.cfg.vad_threshold || old.silence_ms != self.cfg.silence_ms;
        if agent_changed {
            self.restart_agent(ctx);
        }
        if stt_changed {
            self.restart_stt(ctx);
        } else if audio_changed && self.audio.is_some() {
            self.set_listening(false, ctx);
            self.set_listening(true, ctx);
        }
        self.cfg.save();
        self.log("settings applied");
    }

    fn save_geometry(&mut self, ctx: &egui::Context) {
        if self.last_geom_save.elapsed() < Duration::from_secs(2) {
            return;
        }
        self.last_geom_save = Instant::now();
        let (outer, inner) = ctx.input(|i| (i.viewport().outer_rect, i.viewport().inner_rect));
        let zoom = ctx.zoom_factor();
        let (Some(o), Some(inn)) = (outer, inner) else { return };
        // Viewport rects are in points (zoom-scaled); ViewportBuilder expects unzoomed points.
        let pos = [o.min.x * zoom, o.min.y * zoom];
        let size = [inn.width() * zoom, inn.height() * zoom];
        if self.cfg.window_pos != Some(pos) || self.cfg.window_size != size {
            self.cfg.window_pos = Some(pos);
            self.cfg.window_size = size;
            self.draft.window_pos = Some(pos);
            self.draft.window_size = size;
            self.cfg.save();
        }
    }
}

/// The draft differs from the saved settings (window position and size are ignored).
fn same_config(a: &Config, b: &Config) -> bool {
    let strip = |c: &Config| {
        let mut c = c.clone();
        c.window_pos = None;
        c.window_size = [0.0, 0.0];
        serde_json::to_string(&c).unwrap_or_default()
    };
    strip(a) == strip(b)
}

/// Label on the left, control on the right, hint under the label.
fn field(ui: &mut Ui, p: &Pal, label: &str, hint: Option<&str>, add: impl FnOnce(&mut Ui)) {
    let w = ui.available_width();
    ui.horizontal_top(|ui| {
        ui.vertical(|ui| {
            ui.set_width((w * 0.42).max(150.0));
            ui.spacing_mut().item_spacing.y = 2.0;
            ui.label(RichText::new(label).font(theme::reg(14.0)).color(p.text));
            if let Some(h) = hint {
                theme::hint(ui, h, p);
            }
        });
        ui.vertical(|ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
    });
}

fn effort_field(ui: &mut Ui, p: &Pal, s: &S, effort: &mut String) {
    field(ui, p, s.effort, Some(s.effort_hint), |ui| {
        egui::ComboBox::from_id_salt("effort")
            .selected_text(if effort.is_empty() { s.effort_default } else { effort.as_str() })
            .show_ui(ui, |ui| {
                for m in ["", "low", "medium", "high", "xhigh"] {
                    ui.selectable_value(effort, m.to_string(), if m.is_empty() { s.effort_default } else { m });
                }
            });
    });
}

fn key_edit(ui: &mut Ui, p: &Pal, s: &S, key: &mut String, show: &mut bool) {
    ui.horizontal(|ui| {
        let w = ui.available_width() - 56.0;
        ui.add(TextEdit::singleline(key).password(!*show).hint_text("sk-").desired_width(w));
        if theme::button(ui, if *show { s.hide } else { s.show }, Btn::Link, p).clicked() {
            *show = !*show;
        }
    });
}

fn switch_row(ui: &mut Ui, p: &Pal, title: &str, hint: &str, value: &mut bool) -> bool {
    let mut changed = false;
    let w = ui.available_width();
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(w - 56.0);
            ui.spacing_mut().item_spacing.y = 2.0;
            ui.label(RichText::new(title).font(theme::reg(14.0)).color(p.text));
            theme::hint(ui, hint, p);
        });
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            changed = theme::switch(ui, value, p).changed();
        });
    });
    changed
}

/// A status: a dot and quiet text, no capsule. `live` fades the dot once every 1.6 s (only while really listening).
fn chip(ui: &mut Ui, p: &Pal, dot: Color32, live: bool, text: &str) -> egui::Response {
    let font = theme::reg(12.0);
    let g = ui.painter().layout_no_wrap(text.to_string(), font, p.text2);
    let max_w = ui.available_width().max(80.0);
    let w = (g.size().x + 16.0).min(max_w);
    let (r, resp) = ui.allocate_exact_size(vec2(w, 20.0), Sense::hover());
    let dot = if live {
        let t = ui.input(|i| i.time) as f32;
        let k = 0.5 + 0.5 * (t / 1.6 * std::f32::consts::TAU).cos();
        ui.ctx().request_repaint_after(Duration::from_millis(33));
        dot.gamma_multiply(0.4 + 0.6 * k)
    } else {
        dot
    };
    let painter = ui.painter();
    painter.circle_filled(egui::pos2(r.left() + 3.0, r.center().y), 3.0, dot);
    let clip = egui::Rect::from_min_max(egui::pos2(r.left() + 12.0, r.top()), egui::pos2(r.right(), r.bottom()));
    painter.with_clip_rect(clip).galley(egui::pos2(clip.left(), r.center().y - g.size().y / 2.0), g, p.text2);
    resp.on_hover_text(text)
}

// ───────────────────────── icons and icon buttons ─────────────────────────
// 16px box, 1.5px stroke, in the colour of a text role.

#[derive(Clone, Copy, PartialEq)]
enum IconKind {
    /// No fill; a raised step on hover.
    Plain,
    /// The mode is on: raised fill, hairline, accent icon.
    Active,
    /// The primary action of the row: solid accent.
    Filled,
}

fn icon_button(ui: &mut Ui, id: &str, icon: fn(&egui::Painter, egui::Pos2, Color32), kind: IconKind, size: f32, p: &Pal) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(size, size), Sense::click());
    let t = theme::fade(ui, resp.id.with(("hover", id)), resp.hovered());
    let (fill, stroke, col) = match kind {
        IconKind::Plain => (theme::alpha(p.raised, (t * 255.0) as u8), Stroke::NONE, theme::mix(p.text2, p.text, t)),
        IconKind::Active => (theme::mix(p.raised, p.raised_hi, t), Stroke::new(1.0, p.line_hi), p.accent),
        IconKind::Filled => (theme::mix(p.accent, p.accent_hover(), t), Stroke::NONE, p.on_accent),
    };
    let r = if resp.is_pointer_button_down_on() { egui::Rect::from_center_size(rect.center(), rect.size() * 0.98) } else { rect };
    let painter = ui.painter();
    painter.rect(r, theme::R_CONTROL as f32, fill, stroke, egui::StrokeKind::Inside);
    icon(painter, rect.center(), col);
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

fn arc(painter: &egui::Painter, c: egui::Pos2, r: f32, from: f32, to: f32, stroke: Stroke) {
    let pts: Vec<egui::Pos2> = (0..=16)
        .map(|i| {
            let a = from + (to - from) * i as f32 / 16.0;
            c + vec2(a.cos(), a.sin()) * r
        })
        .collect();
    painter.add(egui::Shape::line(pts, stroke));
}

/// The app mark: a headset (the earpiece of the signature, "someone listens").
fn icon_headset(painter: &egui::Painter, c: egui::Pos2, col: Color32) {
    let s = Stroke::new(1.5, col);
    arc(painter, c + vec2(0.0, 1.0), 6.5, std::f32::consts::PI, std::f32::consts::TAU, s);
    for x in [-6.5, 6.5] {
        painter.line_segment([c + vec2(x, 1.0), c + vec2(x, 4.0)], Stroke::new(3.0, col));
    }
    arc(painter, c + vec2(0.0, 4.0), 6.5, 0.2, std::f32::consts::FRAC_PI_2, Stroke::new(1.5, col));
}

fn icon_close(painter: &egui::Painter, c: egui::Pos2, col: Color32) {
    let s = Stroke::new(1.5, col);
    painter.line_segment([c + vec2(-4.5, -4.5), c + vec2(4.5, 4.5)], s);
    painter.line_segment([c + vec2(-4.5, 4.5), c + vec2(4.5, -4.5)], s);
}

fn icon_minus(painter: &egui::Painter, c: egui::Pos2, col: Color32) {
    painter.line_segment([c + vec2(-5.0, 0.0), c + vec2(5.0, 0.0)], Stroke::new(1.5, col));
}

fn icon_gear(painter: &egui::Painter, c: egui::Pos2, col: Color32) {
    let s = Stroke::new(1.5, col);
    painter.circle_stroke(c, 2.5, s);
    for k in 0..8 {
        let a = k as f32 * std::f32::consts::FRAC_PI_4;
        let d = vec2(a.cos(), a.sin());
        painter.line_segment([c + d * 5.0, c + d * 7.0], s);
    }
    painter.circle_stroke(c, 5.0, Stroke::new(1.5, col));
}

fn icon_camera(painter: &egui::Painter, c: egui::Pos2, col: Color32) {
    let s = Stroke::new(1.5, col);
    let body = egui::Rect::from_center_size(c + vec2(0.0, 1.0), vec2(13.0, 9.0));
    painter.rect_stroke(body, 2.0, s, egui::StrokeKind::Middle);
    painter.line_segment([c + vec2(-2.5, -3.5), c + vec2(2.5, -3.5)], s);
    painter.circle_stroke(c + vec2(0.0, 1.0), 2.5, s);
}

fn icon_send(painter: &egui::Painter, c: egui::Pos2, col: Color32) {
    let s = Stroke::new(1.5, col);
    painter.line_segment([c + vec2(-6.0, 0.0), c + vec2(6.0, 0.0)], s);
    painter.line_segment([c + vec2(1.5, -4.5), c + vec2(6.0, 0.0)], s);
    painter.line_segment([c + vec2(1.5, 4.5), c + vec2(6.0, 0.0)], s);
}

/// The OS cursor is visible in screen sharing even though the window is not, so the
/// overlay never changes it: no I-beam / hand / resize shapes, including while
/// drag-selecting text. A plugin added after egui's built-in ones runs its end-of-pass
/// hook last, i.e. after the text-selection plugin that sets the I-beam while dragging.
struct DefaultCursor;

impl egui::plugin::Plugin for DefaultCursor {
    fn debug_name(&self) -> &'static str {
        "default_cursor"
    }
    fn on_end_pass(&mut self, ui: &mut egui::Ui) {
        ui.set_cursor_icon(egui::CursorIcon::Default);
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { s.chars().take(n).collect::<String>() + "…" }
}

impl eframe::App for App {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.updater.is_busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        // A self-update has started the new version: save and close by the normal path.
        if self.updater.should_exit() {
            self.cfg.save();
            ctx.send_viewport_cmd(ViewportCommand::Close);
        }
        if !self.window_ready {
            self.window_ready = win::init_window(&self.title);
            self.capture_excluded = win::is_capture_excluded();
            if self.window_ready {
                self.log(format!("window found, excluded from capture: {}", self.capture_excluded));
            } else {
                ctx.request_repaint_after(Duration::from_millis(50));
            }
        } else if self.last_style_check.elapsed() > Duration::from_secs(1) {
            self.last_style_check = Instant::now();
            win::ensure_tool_window();
            self.capture_excluded = win::is_capture_excluded();
        }
        self.handle_events(ctx);
        self.save_geometry(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let p = Pal::of(ui.visuals().dark_mode);
        let [r, g, b, _] = p.bg.to_array();
        let bg = Color32::from_rgba_unmultiplied(r, g, b, (self.cfg.opacity * 255.0) as u8);
        Frame::new()
            .fill(bg)
            .corner_radius(theme::R_WINDOW)
            .stroke(Stroke::new(1.0, p.line_hi))
            .inner_margin(Margin::same(16))
            .show(ui, |ui| {
                ui.set_min_size(ui.available_size());
                ui.spacing_mut().item_spacing.y = 8.0;
                self.header(ui, &p);
                self.actions(ui, &p);
                self.status_row(ui, &p);
                self.update_banner(ui, &p);
                self.transcript_view(ui, &p);
                self.answers_view(ui, &p);
                self.input_bar(ui, &p);
            });

        // Corner grip for resizing.
        let screen = ctx.content_rect();
        let grip = egui::Rect::from_min_size(screen.max - vec2(18.0, 18.0), vec2(18.0, 18.0));
        let resp = ui.interact(grip, egui::Id::unique("grip"), Sense::drag());
        let painter = ui.painter();
        for k in 0..3 {
            let o = 4.0 + k as f32 * 4.0;
            painter.line_segment([grip.max - vec2(o, 3.0), grip.max - vec2(3.0, o)], Stroke::new(1.5, p.muted));
        }
        if resp.drag_started() {
            ctx.send_viewport_cmd(ViewportCommand::BeginResize(egui::ResizeDirection::SouthEast));
        }

        if self.show_settings {
            self.settings_window(&ctx, &p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accent_is_readable_in_both_themes() {
        for (name, fill, on, surfaces) in [
            ("dark", ACCENT.dark, ACCENT.on_dark, [theme::Pal::dark().sidebar, theme::Pal::dark().bg, theme::Pal::dark().card]),
            ("light", ACCENT.light, ACCENT.on_light, [theme::Pal::light().sidebar, theme::Pal::light().bg, theme::Pal::light().card]),
        ] {
            assert!(theme::contrast(on, fill) >= 4.5, "{name}: on_accent on accent");
            for s in surfaces {
                assert!(theme::contrast(fill, s) >= 4.5, "{name}: accent text on a surface");
            }
        }
    }
}
