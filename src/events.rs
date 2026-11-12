#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hotkey {
    AskScreen,
    AskTranscript,
    ToggleVisible,
    ToggleClickThrough,
    ToggleListen,
    NewChat,
    ScrollUp,
    ScrollDown,
    Monitor(usize),
}

#[derive(Debug)]
pub enum Event {
    Hotkey(Hotkey),
    /// Final transcript of one phrase from the other call participants.
    Transcript(String),
    /// The recognition model is loaded.
    SttReady { model: String, device: String, load_s: String },
    SttStopped,
    AudioLevel(f32),
    AudioError(String),
    /// The agent is ready to take questions.
    AgentReady,
    /// Next chunk of the answer.
    AgentDelta(String),
    /// The answer is complete. `text` is the full text (used when there were no deltas) or the error text.
    AgentDone { text: String, is_error: bool },
    /// The agent process exited; tail of its stderr.
    AgentExited(String),
    /// A question failed; the error is localized by the interface.
    AgentFailed(anyhow::Error),
    Log(String),
}
