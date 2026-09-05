//! Tiny colored, timestamped logging for the server — TabbyAPI-style lines:
//!
//! ```text
//! 2026-09-01 22:19:51.657 INFO:     Received chat completion streaming request …
//! ```
//!
//! Everything goes to stderr. ANSI color is used only when stderr is a TTY and
//! `NO_COLOR` is unset. Use the [`sinfo!`], [`swarn!`], [`serror!`] macros.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COLOR: AtomicBool = AtomicBool::new(false);

/// Detect TTY / `NO_COLOR` once at startup. Safe to call before logging; if it
/// is never called, output is simply never colored.
pub fn init() {
    let color = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    COLOR.store(color, Ordering::Relaxed);
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum Level {
    Info,
    Warn,
    Error,
}

fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as libc::time_t;
    let millis = now.subsec_millis();
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe {
        libc::localtime_r(&secs, &mut tm);
    }
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        millis,
    )
}

/// A value that renders with an ANSI SGR code around it when color is on, and
/// plainly otherwise. Use the [`num`], [`ok`], [`warn_hl`] helpers, or [`paint`].
pub struct Painted<T>(&'static str, T);

impl<T: std::fmt::Display> std::fmt::Display for Painted<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if COLOR.load(Ordering::Relaxed) {
            write!(f, "\x1b[{}m{}\x1b[0m", self.0, self.1)
        } else {
            write!(f, "{}", self.1)
        }
    }
}

/// Wrap a value in an arbitrary SGR code (e.g. `"36"`, `"1;33"`).
pub fn paint<T: std::fmt::Display>(code: &'static str, v: T) -> Painted<T> {
    Painted(code, v)
}

/// Highlight a numeric / metric value (bold cyan).
pub fn num<T: std::fmt::Display>(v: T) -> Painted<T> {
    Painted("1;36", v)
}

/// Highlight a status/quoted string by HTTP status class: 2xx green, 3xx cyan,
/// 4xx yellow, 5xx red, else default.
pub fn status<T: std::fmt::Display>(code: u16, v: T) -> Painted<T> {
    let c = match code / 100 {
        2 => "32",
        3 => "36",
        4 => "33",
        5 => "1;31",
        _ => "0",
    };
    Painted(c, v)
}

#[doc(hidden)]
pub fn log(level: Level, args: std::fmt::Arguments) {
    let (label, ansi) = match level {
        Level::Info => ("INFO", "\x1b[32m"),
        Level::Warn => ("WARNING", "\x1b[33m"),
        Level::Error => ("ERROR", "\x1b[31m"),
    };
    // pad so the message column lines up: "INFO:" + 5 spaces, "WARNING:" + 2 …
    let pad = 10usize.saturating_sub(label.len() + 1);
    let ts = timestamp();
    let mut out = std::io::stderr().lock();
    let _ = if COLOR.load(Ordering::Relaxed) {
        writeln!(
            out,
            "\x1b[2m{ts}\x1b[0m {ansi}{label}:\x1b[0m{:pad$}{args}",
            ""
        )
    } else {
        writeln!(out, "{ts} {label}:{:pad$}{args}", "")
    };
}

#[macro_export]
macro_rules! sinfo {
    ($($a:tt)*) => {
        $crate::server::log::log($crate::server::log::Level::Info, format_args!($($a)*))
    };
}

#[macro_export]
macro_rules! swarn {
    ($($a:tt)*) => {
        $crate::server::log::log($crate::server::log::Level::Warn, format_args!($($a)*))
    };
}

#[macro_export]
macro_rules! serror {
    ($($a:tt)*) => {
        $crate::server::log::log($crate::server::log::Level::Error, format_args!($($a)*))
    };
}
