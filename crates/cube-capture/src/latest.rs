//! The latest frame only: what a camera's thread hands to the one that reads the cube. A reader slower than the
//! camera skips the frames it could not take; it never falls behind.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

struct Slot<T> {
    value: Option<T>,
    dropped: u64,
    error: Option<String>,
}

/// A place for one value, shared between the thread that fills it and the one that empties it
pub struct Latest<T> {
    shared: Arc<(Mutex<Slot<T>>, Condvar)>,
}

impl<T> Clone for Latest<T> {
    fn clone(&self) -> Self {
        Latest { shared: self.shared.clone() }
    }
}

impl<T> Default for Latest<T> {
    fn default() -> Self {
        Latest::new()
    }
}

impl<T> Latest<T> {
    pub fn new() -> Self {
        Latest { shared: Arc::new((Mutex::new(Slot { value: None, dropped: 0, error: None }), Condvar::new())) }
    }

    /// A new value; the one not taken yet is dropped
    pub fn put(&self, value: T) {
        let mut slot = self.shared.0.lock().unwrap();
        slot.dropped += slot.value.replace(value).is_some() as u64;
        self.shared.1.notify_all();
    }

    /// The value, if a new one came since the last take
    pub fn take(&self) -> Option<T> {
        self.shared.0.lock().unwrap().value.take()
    }

    /// The value as soon as there is one, waiting at most `timeout`; None then, or once the source failed
    pub fn wait(&self, timeout: Duration) -> Option<T> {
        let until = Instant::now() + timeout;
        let mut slot = self.shared.0.lock().unwrap();
        loop {
            if slot.value.is_some() || slot.error.is_some() {
                return slot.value.take();
            }
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            slot = self.shared.1.wait_timeout(slot, left).unwrap().0;
        }
    }

    /// Whether a value waits to be taken
    pub fn has_value(&self) -> bool {
        self.shared.0.lock().unwrap().value.is_some()
    }

    /// Values dropped because none took them in time
    pub fn dropped(&self) -> u64 {
        self.shared.0.lock().unwrap().dropped
    }

    /// The source stopped: no value will come any more
    pub fn fail(&self, error: String) {
        self.shared.0.lock().unwrap().error = Some(error);
        self.shared.1.notify_all();
    }

    pub fn error(&self) -> Option<String> {
        self.shared.0.lock().unwrap().error.clone()
    }
}
