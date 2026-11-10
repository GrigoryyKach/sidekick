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
    SttReady(String),
    SttStopped(String),
    AudioLevel(f32),
    AudioError(String),
    /// Агент готов принимать вопросы.
    AgentReady,
    /// Очередной кусок ответа.
    AgentDelta(String),
    /// Ответ закончен. `text` — полный текст (используется, если дельт не было) или текст ошибки.
    AgentDone { text: String, is_error: bool },
    /// Процесс агента завершился; хвост stderr.
    AgentExited(String),
    Log(String),
}
