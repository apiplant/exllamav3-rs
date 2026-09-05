//! Async wrapper for the dynamic-batching [`Generator`] — behavioural port of
//! `generator/async_generator.py` (`AsyncGenerator` / `AsyncJob`, grade —).
//!
//! One cooperative task drives `Generator::iterate()` in a loop and fans each
//! round's [`StreamEvent`]s to the owning job's channel; consumers `await` their
//! [`AsyncJob`] for events. All CUDA work stays on the single OS thread that runs
//! the task (use a `current_thread` runtime + `LocalSet` — `Generator` is not
//! `Sync` and must not migrate threads), exactly as upstream's asyncio model.
//!
//! `AsyncGenerator` [`Deref`]s to the wrapped [`Generator`], so setup calls
//! (`enable_mtp`, `compile_json_schema`, `model()`, …) and the raw
//! `iterate()`/`cancel()` API are available unchanged; the async layer only adds
//! per-job event fan-out (`enqueue` → [`AsyncJob`], [`step`](Self::step)).

use crate::generator::{Generator, JobSpec, StreamEvent};
use anyhow::Result;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use tokio::sync::mpsc;

/// A registered generation job: `await` [`recv`](Self::recv) for its stream of
/// [`StreamEvent`]s until one with `eos == true` (or `None` once the generator
/// drops the job, e.g. on cancellation).
pub struct AsyncJob {
    pub serial: u64,
    rx: mpsc::UnboundedReceiver<StreamEvent>,
}

impl AsyncJob {
    /// Next event for this job, or `None` when the stream is finished.
    pub async fn recv(&mut self) -> Option<StreamEvent> {
        self.rx.recv().await
    }

    /// Non-blocking poll (returns `None` if nothing is ready *or* the job ended).
    pub fn try_recv(&mut self) -> Option<StreamEvent> {
        self.rx.try_recv().ok()
    }
}

pub struct AsyncGenerator {
    gen: Generator,
    /// serial → event sink; dropped when the job emits its eos event
    jobs: HashMap<u64, mpsc::UnboundedSender<StreamEvent>>,
}

impl AsyncGenerator {
    pub fn new(gen: Generator) -> Self {
        Self { gen, jobs: HashMap::new() }
    }

    /// Enqueue a job and return its [`AsyncJob`] handle. Delivery is via an
    /// unbounded channel so a stalled consumer never wedges the shared loop
    /// (upstream issue #227).
    pub fn enqueue_async(&mut self, spec: JobSpec) -> AsyncJob {
        let serial = self.gen.enqueue(spec);
        let (tx, rx) = mpsc::unbounded_channel();
        self.jobs.insert(serial, tx);
        AsyncJob { serial, rx }
    }

    /// Abandon a job: stops it in the generator and drops its event sink so the
    /// consumer's `recv().await` returns `None`.
    pub fn cancel_async(&mut self, serial: u64) {
        self.gen.cancel(serial);
        self.jobs.remove(&serial);
    }

    /// Drive exactly one `iterate()` round, fanning events to the registered
    /// jobs. Returns the number of events dispatched. Call this in a loop from a
    /// `LocalSet` task, yielding (`tokio::task::yield_now().await`) between calls
    /// so consumers and new enqueues can run — mirrors `_run_iteration`.
    pub fn step(&mut self) -> Result<usize> {
        let events = self.gen.iterate()?;
        let n = events.len();
        for ev in events {
            let done = ev.eos;
            let serial = ev.serial;
            if let Some(tx) = self.jobs.get(&serial) {
                let _ = tx.send(ev);
            }
            if done {
                self.jobs.remove(&serial);
            }
        }
        Ok(n)
    }

    /// Anything still queued or generating (across both async and raw jobs).
    pub fn has_work(&self) -> bool {
        self.gen.num_remaining() > 0
    }

    /// Run until every job (async and raw) has finished. Yields to the runtime
    /// between rounds. On an `iterate()` error every live async job's channel is
    /// closed and the error is returned.
    pub async fn run_to_idle(&mut self) -> Result<()> {
        while self.has_work() {
            if let Err(e) = self.step() {
                self.jobs.clear();
                return Err(e);
            }
            tokio::task::yield_now().await;
        }
        Ok(())
    }
}

impl Deref for AsyncGenerator {
    type Target = Generator;
    fn deref(&self) -> &Generator {
        &self.gen
    }
}
impl DerefMut for AsyncGenerator {
    fn deref_mut(&mut self) -> &mut Generator {
        &mut self.gen
    }
}
