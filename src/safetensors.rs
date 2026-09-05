//! Port of `exllamav3/loader/safetensors.py` — header parsing and validation are
//! operation-for-operation identical (grade A). The load path reads each tensor
//! with one positional read into its destination storage, where Python uses the
//! multithreaded C++ `stloader` (wiring). Both avoid demand paging, which is what
//! makes this I/O-bound step track the drive rather than the fault rate.

use anyhow::{bail, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use tch::{Device, Kind, Tensor};

const MAX_HEADER_SIZE: u64 = 100 * 1024 * 1024;

/// Below this a tensor is read by one thread: the split stops paying for itself
/// once each request is small enough that per-read overhead dominates.
const MIN_PARALLEL_CHUNK: usize = 2 * 1024 * 1024;
/// Past ~8 concurrent readers the drive is already saturated on this hardware.
const MAX_READ_THREADS: usize = 8;

/// Load-path counters, so the disk and PCIe halves of a load can be attributed
/// separately instead of guessed at. Read with [`load_prof_take`].
pub static READ_NS: AtomicU64 = AtomicU64::new(0);
pub static H2D_NS: AtomicU64 = AtomicU64::new(0);
pub static READ_BYTES: AtomicU64 = AtomicU64::new(0);

/// `(read thread-seconds, h2d seconds, bytes)` since the last call, resetting
/// them. Read time is summed over the reader threads, so with prefetching on it
/// exceeds wall-clock — that gap is the overlap the pipeline is buying.
pub fn load_prof_take() -> (f64, f64, u64) {
    (
        READ_NS.swap(0, Ordering::Relaxed) as f64 / 1e9,
        H2D_NS.swap(0, Ordering::Relaxed) as f64 / 1e9,
        READ_BYTES.swap(0, Ordering::Relaxed),
    )
}

/// (torch kind, element size in bytes). Mirrors `convert_dtype`.
fn convert_dtype(dt: &str) -> Result<(Kind, usize)> {
    Ok(match dt {
        "I32" => (Kind::Int, 4),
        "I64" => (Kind::Int64, 8),
        "I8" => (Kind::Int8, 1),
        "F8_E8M0" => (Kind::Uint8, 1),
        "I16" => (Kind::Int16, 2),
        "F16" => (Kind::Half, 2),
        "BF16" => (Kind::BFloat16, 2),
        "F32" => (Kind::Float, 4),
        "F8_E4M3" => (Kind::Float8e4m3fn, 1),
        "U8" => (Kind::Uint8, 1),
        _ => bail!("Unknown dtype {dt}"),
    })
}

#[derive(Clone)]
struct Entry {
    dtype: String,
    shape: Vec<i64>,
    begin: u64,
    end: u64,
}

/// Faithful port of `validate_header`: every attacker-controllable field is
/// checked before it can size an allocation or seek.
fn validate_header(
    header: &serde_json::Map<String, serde_json::Value>,
    filename: &str,
    data_offset: u64,
    file_size: u64,
) -> Result<()> {
    for (key, h) in header {
        if key == "__metadata__" {
            continue;
        }
        let h = h
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid entry for {key} in {filename}"))?;
        let shape: Vec<i64> = h
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid shape for {key} in {filename}"))?
            .iter()
            .map(|v| v.as_i64().filter(|x| *x >= 0))
            .collect::<Option<_>>()
            .ok_or_else(|| anyhow::anyhow!("Invalid shape for {key} in {filename}"))?;
        let offs = h
            .get("data_offsets")
            .and_then(|s| s.as_array())
            .filter(|a| a.len() == 2)
            .ok_or_else(|| anyhow::anyhow!("Invalid data_offsets for {key} in {filename}"))?;
        let beg = offs[0].as_u64();
        let end = offs[1].as_u64();
        let (beg, end) = match (beg, end) {
            (Some(b), Some(e)) if b <= e => (b, e),
            _ => bail!("Invalid data_offsets for {key} in {filename}"),
        };
        if data_offset + end > file_size {
            bail!("Tensor {key} in {filename} extends past end of file");
        }
        let dt = h.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
        if let Ok((_, esize)) = convert_dtype(dt) {
            let expected: i64 = shape.iter().product::<i64>() * esize as i64;
            if (end - beg) as i64 != expected {
                bail!("Size mismatch for {key} in {filename}: span {} vs {expected}", end - beg);
            }
        }
    }
    Ok(())
}

fn read_header(path: &Path, fixes: &[(String, String)]) -> Result<HashMap<String, Entry>> {
    use std::io::Read;

    // Read only the header, never the tensor payload. Slurping the whole file to
    // parse its first few KB costs a full pass over every shard (tens of GB) that
    // is then thrown away, and evicts the page cache the mmap load is about to want.
    let mut file = File::open(path)?;
    let filename = path.display().to_string();
    let file_len = file.metadata()?.len();
    if file_len < 8 {
        bail!("{filename} too small to be a safetensors file");
    }
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_size = u64::from_le_bytes(len_buf);
    if !(0 < header_size && header_size <= MAX_HEADER_SIZE) {
        bail!("Invalid safetensors header size in {filename}: {header_size}");
    }
    if 8 + header_size > file_len {
        bail!("Truncated safetensors header in {filename}");
    }
    let mut header_buf = vec![0u8; header_size as usize];
    file.read_exact(&mut header_buf)?;
    let json: serde_json::Value = serde_json::from_slice(&header_buf)?;
    let obj = json
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid safetensors header in {filename}"))?;
    let data_offset = 8 + header_size;
    validate_header(obj, &filename, data_offset, file_len)?;

    let mut out = HashMap::new();
    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        let mut key = k.clone();
        for (suf, rep) in fixes {
            if key.ends_with(suf.as_str()) {
                key = format!("{}{}", &key[..key.len() - suf.len()], rep);
            }
        }
        let h = v.as_object().unwrap();
        let offs = h["data_offsets"].as_array().unwrap();
        out.insert(
            key,
            Entry {
                dtype: h["dtype"].as_str().unwrap().to_string(),
                shape: h["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect(),
                begin: data_offset + offs[0].as_u64().unwrap(),
                end: data_offset + offs[1].as_u64().unwrap(),
            },
        );
    }
    Ok(out)
}

/// Cap on prefetched-but-unclaimed host bytes. Readers park once they are this
/// far ahead, so the pipeline cannot outrun the consumer into swap.
const PREFETCH_CAP_BYTES: u64 = 2 << 30;
/// Concurrent tensor reads. Several tensors in flight is what keeps the drive's
/// queue deep across the small ones, where a single tensor cannot fill it.
const PREFETCH_WORKERS: usize = 4;

/// The file handles and header index, shared with the prefetch workers.
struct Shared {
    files: Vec<File>,
    map: HashMap<String, (usize, Entry)>, // key -> (file idx, entry)
}

enum Slot {
    /// Not started; whoever gets here first reads it.
    Pending,
    /// A worker is reading it — a consumer waits rather than reading it twice.
    InFlight,
    Ready(Tensor, u64),
    /// Claimed by a consumer that read it itself; workers must skip it.
    Taken,
}

struct PrefetchInner {
    slots: HashMap<String, Slot>,
    /// Keys in file-offset order. The checkpoint stores tensors in load order
    /// (embeddings, then layers 0..N), and the trunk's keys occupy one
    /// contiguous run at the front, so producing in this order tracks the
    /// consumer and the reads stay sequential on the drive. Another instance's
    /// vision/MTP keys sit past the end of that run and are only reached once
    /// the trunk is done, where they harmlessly fill the cap.
    queue: Vec<String>,
    next: usize,
    ready_bytes: u64,
    stop: bool,
}

struct Prefetch {
    inner: Mutex<PrefetchInner>,
    cv: Condvar,
}

pub struct SafeTensors {
    shared: Arc<Shared>,
    dir: PathBuf,
    prefetch: Option<Arc<Prefetch>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// Allocate the destination tensor and fill it with one positional read.
fn read_tensor(shared: &Shared, idx: usize, e: &Entry) -> Result<Tensor> {
    use std::os::unix::fs::FileExt;
    let (kind, _esize) = convert_dtype(&e.dtype)?;
    let nbytes = (e.end - e.begin) as usize;

    // Read straight into the destination tensor's storage. The obvious
    // alternative — mmap the shard and slice it — reads the file through demand
    // paging, which on NVMe means synchronous 128 KB faults on one thread and
    // leaves the drive's queue almost empty; a positional read of the whole
    // tensor at once is one syscall over megabytes.
    let cpu = Tensor::empty(&e.shape, (kind, Device::Cpu));
    let t_read = std::time::Instant::now();
    {
        // `cpu` was just allocated contiguous with exactly `nbytes` of storage,
        // and nothing else aliases it.
        let dst = unsafe { std::slice::from_raw_parts_mut(cpu.data_ptr() as *mut u8, nbytes) };
        let file = &shared.files[idx];
        // One synchronous read keeps only one request in flight, which leaves an
        // NVMe drive idle between completions; splitting a large tensor across a
        // few threads keeps the queue deep. Measured on this model's shards:
        // ~1.9 GB/s single-threaded vs ~3.0 GB/s at 8, against a 4.0 GB/s device.
        let nthreads = (nbytes / MIN_PARALLEL_CHUNK).clamp(1, MAX_READ_THREADS);
        if nthreads <= 1 {
            file.read_exact_at(dst, e.begin)?;
        } else {
            let per = nbytes.div_ceil(nthreads);
            std::thread::scope(|s| -> Result<()> {
                let mut handles = Vec::with_capacity(nthreads);
                for (i, part) in dst.chunks_mut(per).enumerate() {
                    let off = e.begin + (i * per) as u64;
                    handles.push(s.spawn(move || file.read_exact_at(part, off)));
                }
                for h in handles {
                    h.join().map_err(|_| anyhow::anyhow!("reader thread panicked"))??;
                }
                Ok(())
            })?;
        }
    }
    READ_NS.fetch_add(t_read.elapsed().as_nanos() as u64, Ordering::Relaxed);
    READ_BYTES.fetch_add(nbytes as u64, Ordering::Relaxed);
    Ok(cpu)
}

impl SafeTensors {
    pub fn open(dir: &Path, fixes: &[(String, String)]) -> Result<Self> {
        let mut handles = Vec::new();
        let mut map = HashMap::new();
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no *.safetensors in {}", dir.display());
        }
        for f in files {
            let header = read_header(&f, fixes)?;
            let idx = handles.len();
            handles.push(File::open(&f)?);
            for (k, e) in header {
                map.insert(k, (idx, e));
            }
        }
        Ok(Self {
            shared: Arc::new(Shared { files: handles, map }),
            dir: dir.to_path_buf(),
            prefetch: None,
            workers: Vec::new(),
        })
    }

    /// Start reading tensors ahead of demand on background threads.
    ///
    /// Loading is disk-bound, and reading one tensor at a time leaves the drive
    /// idle during every dequant, H2D copy and module construction in between —
    /// measured at 2.8 GB/s against a 4.0 GB/s device, and collapsing to 1.3 GB/s
    /// when the machine is under memory pressure. Reading ahead overlaps the disk
    /// with all of that and keeps several requests in flight at once.
    ///
    /// Opt-in because several `SafeTensors` over one directory each read a
    /// disjoint subset (trunk / vision / MTP); only the trunk's is worth
    /// pipelining, and having all of them prefetch would read the file repeatedly.
    pub fn start_prefetch(&mut self) {
        if self.prefetch.is_some() || std::env::var("EXL3_NO_PREFETCH").is_ok() {
            return;
        }
        let mut queue: Vec<String> = self.shared.map.keys().cloned().collect();
        queue.sort_by_key(|k| {
            let (idx, e) = &self.shared.map[k];
            (*idx, e.begin)
        });
        let slots = queue.iter().map(|k| (k.clone(), Slot::Pending)).collect();
        let pf = Arc::new(Prefetch {
            inner: Mutex::new(PrefetchInner {
                slots,
                queue,
                next: 0,
                ready_bytes: 0,
                stop: false,
            }),
            cv: Condvar::new(),
        });
        for _ in 0..PREFETCH_WORKERS {
            let pf = Arc::clone(&pf);
            let shared = Arc::clone(&self.shared);
            self.workers.push(std::thread::spawn(move || prefetch_worker(&pf, &shared)));
        }
        self.prefetch = Some(pf);
    }

    pub fn has(&self, key: &str) -> bool {
        self.shared.map.contains_key(key)
    }

    /// Hand back a prefetched tensor, or read it here if the pipeline has not
    /// reached it yet. A consumer never blocks waiting for a worker to *start*
    /// something — only for one already mid-read — so it can always overtake the
    /// pipeline and cannot deadlock against the byte cap.
    fn take_or_read(&self, key: &str, idx: usize, e: &Entry) -> Result<Tensor> {
        let Some(pf) = &self.prefetch else {
            return read_tensor(&self.shared, idx, e);
        };
        let mut inner = pf.inner.lock().unwrap();
        loop {
            match inner.slots.get(key) {
                Some(Slot::Ready(..)) => {
                    let Some(Slot::Ready(t, n)) = inner.slots.insert(key.to_string(), Slot::Taken)
                    else {
                        unreachable!("checked Ready above while holding the lock")
                    };
                    inner.ready_bytes -= n;
                    pf.cv.notify_all(); // a worker may be parked on the cap
                    return Ok(t);
                }
                Some(Slot::InFlight) => {
                    inner = pf.cv.wait(inner).unwrap();
                }
                _ => {
                    // Pending or unknown: claim it so no worker duplicates the
                    // read, then do it ourselves outside the lock.
                    inner.slots.insert(key.to_string(), Slot::Taken);
                    drop(inner);
                    return read_tensor(&self.shared, idx, e);
                }
            }
        }
    }

    pub fn has_group(&self, key: &str, subkeys: &[&str]) -> bool {
        subkeys.iter().all(|s| self.has(&format!("{key}.{s}")))
    }

    /// `get_tensor` — loads to `device`. `float2half` casts F32→F16, and BF16→F16
    /// unless `allow_bf16` (matching the Python defaults).
    pub fn get(
        &self,
        key: &str,
        device: Device,
        float2half: bool,
        allow_bf16: bool,
    ) -> Result<Tensor> {
        let (idx, e) = self
            .shared
            .map
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("Required tensor {key} not found in {}", self.dir.display()))?;
        let cpu = self.take_or_read(key, *idx, e)?;

        let t_h2d = std::time::Instant::now();
        let mut t = cpu.to_device(device);
        if t.kind() == Kind::BFloat16 && !allow_bf16 {
            t = t.to_kind(Kind::Half);
        }
        if t.kind() == Kind::Float && float2half {
            t = t.to_kind(Kind::Half);
        }
        let out = t.contiguous();
        H2D_NS.fetch_add(t_h2d.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(out)
    }

    pub fn get_opt(&self, key: &str, device: Device) -> Option<Tensor> {
        if self.has(key) {
            self.get(key, device, false, false).ok()
        } else {
            None
        }
    }
}

/// Read queued tensors until the queue drains or the owner drops us.
fn prefetch_worker(pf: &Prefetch, shared: &Shared) {
    loop {
        let (key, idx, entry) = {
            let mut inner = pf.inner.lock().unwrap();
            // Park while far enough ahead of the consumer.
            while inner.ready_bytes >= PREFETCH_CAP_BYTES && !inner.stop {
                inner = pf.cv.wait(inner).unwrap();
            }
            if inner.stop {
                return;
            }
            // Next key nobody has claimed.
            let mut found = None;
            while inner.next < inner.queue.len() {
                let k = inner.queue[inner.next].clone();
                inner.next += 1;
                if matches!(inner.slots.get(&k), Some(Slot::Pending)) {
                    inner.slots.insert(k.clone(), Slot::InFlight);
                    found = Some(k);
                    break;
                }
            }
            let Some(k) = found else { return };
            let (idx, e) = shared.map[&k].clone();
            (k, idx, e)
        };

        let n = entry.end - entry.begin;
        let read = read_tensor(shared, idx, &entry);
        let mut inner = pf.inner.lock().unwrap();
        match read {
            Ok(t) => {
                inner.slots.insert(key, Slot::Ready(t, n));
                inner.ready_bytes += n;
            }
            // Leave it Pending so the consumer retries inline and surfaces the
            // real error at the call site that actually needs the tensor.
            Err(_) => {
                inner.slots.insert(key, Slot::Pending);
            }
        }
        pf.cv.notify_all();
    }
}

impl Drop for SafeTensors {
    fn drop(&mut self) {
        if let Some(pf) = &self.prefetch {
            pf.inner.lock().unwrap().stop = true;
            pf.cv.notify_all();
        }
        for w in self.workers.drain(..) {
            w.join().ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check the read path against a reference implementation. The offsets
    /// and the parallel split are the kind of arithmetic that fails silently —
    /// a wrong `begin` still yields a well-formed tensor of the right shape — so
    /// this dumps checksums for `tests/safetensors_check.py` to compare against
    /// Python's `safetensors`. Inert unless `EXL3_ST_MODEL` points at a model.
    #[test]
    fn dump_tensor_checksums() {
        let Ok(dir) = std::env::var("EXL3_ST_MODEL") else { return };
        let out = std::env::var("EXL3_ST_OUT").expect("EXL3_ST_OUT");
        let mut st = SafeTensors::open(Path::new(&dir), &[]).unwrap();
        // Exercise the prefetch path, so the differential check covers the
        // pipeline the real load actually uses and not just the inline reader.
        st.start_prefetch();
        let mut keys: Vec<_> = st.shared.map.keys().cloned().collect();
        keys.sort();
        let mut rows = serde_json::Map::new();
        for k in keys {
            // raw bytes, no dtype coercion, so the comparison is byte-exact
            let t = st.get(&k, Device::Cpu, false, true).unwrap();
            let n = {
                let (_, e) = &st.shared.map[&k];
                (e.end - e.begin) as usize
            };
            let bytes =
                unsafe { std::slice::from_raw_parts(t.data_ptr() as *const u8, n) };
            // Sampled at a stride rather than hashed whole: the reference side is
            // Python, and 17 GB of byte-at-a-time hashing there is not worth it.
            // Every parallel chunk is >=1 MB, so a 4 KB stride still lands hundreds
            // of samples inside each one and in the bytes either side of every seam.
            let mut h: u64 = 1469598103934665603;
            h = (h ^ n as u64).wrapping_mul(1099511628211);
            for b in bytes.iter().step_by(4096) {
                h = (h ^ *b as u64).wrapping_mul(1099511628211);
            }
            rows.insert(k, serde_json::json!(format!("{h:016x}")));
        }
        std::fs::write(out, serde_json::to_string(&rows).unwrap()).unwrap();
    }
}
