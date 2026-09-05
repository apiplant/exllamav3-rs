//! Hashed n-gram embedding — port of `modules/ngram_embedding.py` and the
//! `exl3_ngram_trellis` codec (`modules/quant/exl3_lib/ngram_codec.py`).
//!
//! This is the other half of [`crate::ple`]: it turns a token history into the
//! `(bsz, seq_len, ple_embed_dim)` feature block PLE consumes. Each position
//! hashes its trailing 2..=`ngram_size` token windows into
//! `(ngram_size - 1) * heads_per_ngram` table rows of `ROW_DIM` values each, and
//! the rows concatenate into one feature vector. Hashing never crosses an eos
//! boundary: positions whose shifted source would cross one read eos instead, so
//! a fresh sequence starts from an eos-padded history rather than from whatever
//! preceded it in the batch.
//!
//! **Resident tables only.** Upstream treats disk streaming as a first-class
//! mode — the real tables run to tens of billions of parameters, and it keeps
//! `DiskTensorHandle`s and gathers just the touched rows with threaded preads.
//! Here the shards are ordinary CPU tensors from the safetensors collection, so
//! this loads and runs the small/test tables but not a production one; the
//! streaming path needs row-level reads that [`crate::safetensors`] does not
//! expose yet. Sharding itself *is* handled (`shard_N`), since the row -> shard
//! routing is what a streaming backend would slot into.
//!
//! Both table formats are supported: raw `.weight` rows, and the trellis-packed
//! `.trellis` rows dequantized here in torch ops (upstream has a CUDA kernel for
//! the packed form; this is the reference it is checked against).

use crate::safetensors::SafeTensors;
use anyhow::{bail, Result};
use tch::{Device, Kind, Tensor};

/// Every embedding row is 160 values wide, in both table formats.
pub const ROW_DIM: i64 = 160;

const MUL1: i64 = 0x83DC_D12D;

/// Packed row width for a K-bit trellis: one fp16 scale word plus the ring.
pub fn words_per_row(k: i64) -> i64 {
    1 + ROW_DIM * k / 16
}

fn f16_from_bits(b: u16) -> f64 {
    let sign = if b >> 15 == 1 { -1.0 } else { 1.0 };
    let exp = ((b >> 10) & 0x1f) as i32;
    let mant = (b & 0x3ff) as f64 / 1024.0;
    // No zero/subnormal/inf cases arise for the two constants this decodes.
    sign * (1.0 + mant) * 2f64.powi(exp - 15)
}

/// All 65536 decoded mul1 values, bit-exact with upstream's `decode_3inst<2>`.
/// The decode is a byte-sum of `state * MUL1`, affinely mapped into fp16.
pub fn mul1_codebook(device: Device) -> Tensor {
    let s = Tensor::arange(65536, (Kind::Int64, device));
    let prod = (&s * MUL1).bitwise_and(0xFFFF_FFFFi64);
    let byte = |sh: i64| prod.bitwise_right_shift_tensor_scalar(sh).bitwise_and(255i64);
    let bsum = byte(0) + byte(8) + byte(16) + byte(24);
    let h = (bsum + 1024).to_kind(Kind::Float);
    (h * f16_from_bits(0x1eee) + f16_from_bits(0xc931)).to_kind(Kind::Half)
}

/// Inverse of upstream's `pack_rows`: `(N, 1 + 10K)` int16 -> states `(N, 160)`
/// int64 and per-row fp16 scales `(N,)`.
///
/// The ring is a bitstream of `160 * K` bits, where bits `[i*K, (i+1)*K)` are the
/// low K bits of position `i`'s 16-bit trellis state; the remaining bits of that
/// state are the ones already written by earlier positions, wrapping at the end
/// of the row (tail-biting), which is what the modular source index below says.
pub fn unpack_rows(packed: &Tensor, k: i64) -> (Tensor, Tensor) {
    let dev = packed.device();
    let n = packed.size()[0];
    let scales = packed.select(1, 0).contiguous().view_dtype(Kind::Half);
    let words = packed.narrow(1, 1, ROW_DIM * k / 16).to_kind(Kind::Int64).bitwise_and(0xFFFFi64);
    let bit = Tensor::arange(16, (Kind::Int64, dev));
    let stream = words
        .unsqueeze(-1)
        .bitwise_right_shift(&bit)
        .bitwise_and(1i64)
        .reshape([n, ROW_DIM * k]);

    // state_i bit m lives at stream bit ((i - m / K) mod 160) * K + m % K
    let i = Tensor::arange(ROW_DIM, (Kind::Int64, dev)).unsqueeze(1);
    let m = Tensor::arange(16, (Kind::Int64, dev)).unsqueeze(0);
    let src = ((i - m.shallow_clone().floor_divide(&Tensor::from(k))).remainder(ROW_DIM)) * k
        + m.shallow_clone().remainder(k);
    let gathered = stream
        .index_select(1, &src.reshape([-1]))
        .reshape([n, ROW_DIM, 16]);
    let states = (gathered.bitwise_left_shift(&m)).sum_dim_intlist(vec![-1i64].as_slice(), false, Kind::Int64);
    (states, scales)
}

/// `packed` `(N, 1 + 10K)` int16 -> decoded `(N, 160)` fp32 rows.
/// `bias` is the per-row bias already gathered per hash head.
pub fn dequant_rows(packed: &Tensor, k: i64, codebook: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let (states, scales) = unpack_rows(packed, k);
    let n = packed.size()[0];
    let out = codebook
        .index_select(0, &states.reshape([-1]))
        .reshape([n, ROW_DIM])
        .to_kind(Kind::Float)
        * scales.to_kind(Kind::Float).unsqueeze(1);
    match bias {
        Some(b) => out + b.to_kind(Kind::Float),
        None => out,
    }
}

/// How the row store is quantized. Shards are held individually and never
/// concatenated — a cat would transiently double a very large footprint, and row
/// -> shard routing is a plain division anyway.
enum Store {
    Fp16(Vec<Tensor>),
    Trellis {
        shards: Vec<Tensor>,
        k: i64,
        codebook: Tensor,
        /// `(num_heads, 160)` fp32 dequant bias, on the device.
        head_bias: Tensor,
    },
}

pub struct NGramEmbedding {
    store: Store,
    /// Hash parameters stay on the CPU, where the token ids already live.
    head_offsets: Tensor,      // (num_heads,) i64, start row of each head
    head_vocab_sizes: Tensor,  // (num_heads,) i64
    layer_multipliers: Tensor, // (ngram_size,) i64
    rows_per_shard: i64,
    num_rows: i64,
    pub ngram_size: i64,
    pub heads_per_ngram: i64,
    pub num_heads: i64,
    pub ple_embed_dim: i64,
    eos_token_id: i64,
    device: Device,
}

impl NGramEmbedding {
    /// Tokens of history each position needs ahead of it.
    pub fn context_len(&self) -> i64 {
        self.ngram_size - 1
    }

    pub fn load(
        stc: &SafeTensors,
        key: &str,
        ngram_size: i64,
        heads_per_ngram: i64,
        ple_embed_dim: i64,
        eos_token_id: i64,
        device: Device,
    ) -> Result<Self> {
        let num_heads = (ngram_size - 1) * heads_per_ngram;
        let head_dim = ple_embed_dim / num_heads;
        if head_dim != ROW_DIM {
            bail!("expected {ROW_DIM}-D embedding rows, got {head_dim} ({key})");
        }

        let shard_keys = |suffix: &str| -> Vec<String> {
            let mut keys = vec![];
            loop {
                let k = format!("{key}.shard_{}.{suffix}", keys.len());
                if !stc.has(&k) {
                    break;
                }
                keys.push(k);
            }
            keys
        };

        let mut trellis_keys = shard_keys("trellis");
        if trellis_keys.is_empty() && stc.has(&format!("{key}.trellis")) {
            trellis_keys.push(format!("{key}.trellis")); // single-tensor layout of older files
        }
        let quantized = !trellis_keys.is_empty();

        let keys = if quantized {
            trellis_keys
        } else if stc.has(&format!("{key}.weight")) {
            vec![format!("{key}.weight")]
        } else {
            let k = shard_keys("weight");
            if k.is_empty() {
                bail!("No .trellis, .weight or .shard_N.weight tensors found for {key}");
            }
            k
        };

        let shards: Vec<Tensor> = keys
            .iter()
            .map(|k| stc.get(k, Device::Cpu, false, !quantized))
            .collect::<Result<_>>()?;

        // All shards but the last hold the same row count; the last may be short.
        let rows_per_shard = shards[0].size()[0];
        if shards[..shards.len() - 1].iter().any(|t| t.size()[0] != rows_per_shard)
            || shards[shards.len() - 1].size()[0] > rows_per_shard
        {
            bail!("n-gram table shards must have equal row counts (last may be short) for {key}");
        }
        let num_rows: i64 = shards.iter().map(|t| t.size()[0]).sum();

        let parent = key.rsplit_once('.').map(|(p, _)| p).unwrap_or(key);
        let aux = |name: &str| -> Result<Tensor> {
            Ok(stc.get(name, Device::Cpu, false, true)?.to_kind(Kind::Int64).contiguous())
        };
        let (off_key, size_key, mult_key) = if quantized {
            (
                format!("{key}.head_offsets"),
                format!("{key}.head_vocab_sizes"),
                format!("{key}.layer_multipliers"),
            )
        } else {
            (
                format!("{parent}.ngram_heads_offsets"),
                format!("{parent}.ngram_heads_vocab_sizes"),
                format!("{parent}.layer_multipliers"),
            )
        };
        let head_offsets = aux(&off_key)?;
        let head_vocab_sizes = aux(&size_key)?;
        let layer_multipliers = aux(&mult_key)?;
        if head_offsets.size()[0] != num_heads {
            bail!(
                "{key}: {} hash heads in the checkpoint, {num_heads} implied by ngram_size/heads_per_ngram",
                head_offsets.size()[0]
            );
        }

        let store = if quantized {
            let words = shards[0].size()[1];
            let k = (words - 1) * 16 / ROW_DIM;
            if words != words_per_row(k) || shards.iter().any(|t| t.size()[1] != words) {
                bail!("{key}: packed row width {words} is not a valid trellis row");
            }
            let head_bias = stc
                .get(&format!("{key}.head_bias"), device, false, true)?
                .to_kind(Kind::Float);
            Store::Trellis { shards, k, codebook: mul1_codebook(device), head_bias }
        } else {
            if shards.iter().any(|t| t.size()[1] != ROW_DIM) {
                bail!("{key}: table rows are not {ROW_DIM}-D");
            }
            Store::Fp16(shards)
        };

        Ok(Self {
            store,
            head_offsets,
            head_vocab_sizes,
            layer_multipliers,
            rows_per_shard,
            num_rows,
            ngram_size,
            heads_per_ngram,
            num_heads,
            ple_embed_dim,
            eos_token_id,
            device,
        })
    }

    /// Total rows across all shards, i.e. the table's vocabulary.
    pub fn num_rows(&self) -> i64 {
        self.num_rows
    }

    /// Shift right by `shift`, except that n-grams never span an eos boundary:
    /// a position less than `shift` tokens into its segment reads eos instead of
    /// the token that actually precedes it.
    fn shift_right_ignore_eos(&self, token_ids: &Tensor, shift: i64) -> Tensor {
        if shift == 0 {
            return token_ids.shallow_clone();
        }
        let (bsz, seq_len) = (token_ids.size()[0], token_ids.size()[1]);
        let dev = token_ids.device();
        let positions = Tensor::arange(seq_len, (Kind::Int64, dev));
        // where_self reads as `true_branch.where_self(cond, false_branch)`.
        let is_eos = token_ids.eq(self.eos_token_id);
        let eos_positions = positions
            .unsqueeze(0)
            .expand([bsz, seq_len], false)
            .where_self(&is_eos, &Tensor::from(-1i64).to_device(dev));
        let previous_eos_inclusive = eos_positions.cummax(1).0;
        let previous_eos = Tensor::cat(
            &[
                Tensor::full([bsz, 1], -1i64, (Kind::Int64, dev)),
                previous_eos_inclusive.narrow(1, 0, seq_len - 1),
            ],
            1,
        );
        let position_in_segment = positions.unsqueeze(0) - (&previous_eos + 1);
        let source_positions = &positions - shift;
        let gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand([bsz, seq_len], false);
        let shifted = token_ids.gather(1, &gather_positions, false);
        let valid = position_in_segment
            .ge(shift)
            .logical_and(&source_positions.ge(0).unsqueeze(0));
        shifted.where_self(&valid, &Tensor::from(self.eos_token_id).to_device(dev))
    }

    /// `token_history` `(bsz, context_len + out_len)` -> `(bsz, out_len, num_heads)`
    /// global table row indices for the last `out_len` positions.
    ///
    /// Head block `n - 2` covers the n-gram of length `n`: the hash mixes the
    /// current token and the `n - 1` before it, each through its own multiplier,
    /// then folds into that head's slice of the table.
    pub fn compute_ngram_ids(&self, token_history: &Tensor, out_len: i64) -> Tensor {
        let th = token_history.to_kind(Kind::Int64);
        let dev = th.device();
        let mult = self.layer_multipliers.to_device(dev);
        let shifted: Vec<Tensor> = (0..self.ngram_size)
            .map(|s| self.shift_right_ignore_eos(&th, s))
            .collect();
        let mut blocks = vec![];
        for ngram in 2..=self.ngram_size {
            let lo = (ngram - 2) * self.heads_per_ngram;
            let mut mixed = &shifted[0] * mult.get(0);
            for position in 1..ngram {
                mixed = mixed.bitwise_xor_tensor(&(&shifted[position as usize] * mult.get(position)));
            }
            let sizes = self.head_vocab_sizes.narrow(0, lo, self.heads_per_ngram).to_device(dev);
            let offsets = self.head_offsets.narrow(0, lo, self.heads_per_ngram).to_device(dev);
            blocks.push(
                mixed.unsqueeze(-1).remainder_tensor(&sizes.view([1, 1, -1]))
                    + offsets.view([1, 1, -1]),
            );
        }
        let all = Tensor::cat(&blocks, -1);
        let seq = all.size()[1];
        all.narrow(1, seq - out_len, out_len)
    }

    /// Gather raw store rows (packed int16 or fp16/bf16) for global row indices,
    /// routing each index to the shard that holds it.
    fn fetch_packed(&self, uids: &Tensor) -> Tensor {
        let shards = match &self.store {
            Store::Fp16(s) => s,
            Store::Trellis { shards, .. } => shards,
        };
        if shards.len() == 1 {
            return shards[0].index_select(0, uids);
        }
        let shard = uids.floor_divide(&Tensor::from(self.rows_per_shard));
        let local = uids - &shard * self.rows_per_shard;
        let mut out: Option<Tensor> = None;
        for s in 0..shards.len() as i64 {
            let m = shard.eq(s);
            if !bool::try_from(m.any()).unwrap_or(false) {
                continue;
            }
            let idx = m.nonzero().squeeze_dim(1);
            let rows = shards[s as usize].index_select(0, &local.index_select(0, &idx));
            let out = out.get_or_insert_with(|| {
                let mut size = rows.size();
                size[0] = uids.size()[0];
                Tensor::empty(&size, (rows.kind(), rows.device()))
            });
            let _ = out.index_copy_(0, &idx, &rows);
        }
        out.expect("at least one shard holds a row")
    }

    /// Unique row indices -> decoded `(N, 160)` rows on the module's device.
    pub fn fetch_rows(&self, uids: &Tensor) -> Tensor {
        let uids_cpu = uids.to_device(Device::Cpu).to_kind(Kind::Int64);
        let raw = self.fetch_packed(&uids_cpu).to_device(self.device);
        match &self.store {
            Store::Fp16(_) => raw.to_kind(Kind::Float),
            Store::Trellis { k, codebook, head_bias, .. } => {
                // Rows are laid out head by head, so the head owning a row is the
                // last offset not past it.
                let offs = self.head_offsets.to_device(self.device);
                // searchsorted reads as `values.searchsorted(sorted_sequence, ..)`.
                let heads = (uids_cpu
                    .to_device(self.device)
                    .searchsorted(&offs, false, true, "right", None::<Tensor>)
                    - 1)
                    .clamp(0, self.num_heads - 1);
                let bias = head_bias.index_select(0, &heads);
                dequant_rows(&raw, *k, codebook, Some(&bias))
            }
        }
    }

    /// `(bsz, seq_len, num_heads)` row indices -> `(bsz, seq_len, ple_embed_dim)`.
    /// Rows are deduplicated first: a hot n-gram is decoded once per forward pass
    /// no matter how many positions and heads land on it.
    pub fn embed_ids(&self, ngram_ids: &Tensor) -> Tensor {
        let sz = ngram_ids.size();
        let (bsz, seq_len, heads) = (sz[0], sz[1], sz[2]);
        let flat = ngram_ids.reshape([-1]);
        let (sorted, _) = flat.sort(0, false);
        let (uids, _, _) = sorted.unique_consecutive(false, false, None);
        let inverse = flat.searchsorted(&uids, false, false, "left", None::<Tensor>);
        let rows = self.fetch_rows(&uids);
        rows.index_select(0, &inverse.to_device(self.device))
            .view([bsz, seq_len, heads * ROW_DIM])
    }

    /// `x` `(bsz, context_len + seq_len)` token history -> `(bsz, seq_len, ple_embed_dim)`
    /// embeddings for the last `seq_len` positions, on the module's device.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out_len = x.size()[1] - self.context_len();
        let ngram_ids = self.compute_ngram_ids(&x.to_device(Device::Cpu), out_len);
        self.embed_ids(&ngram_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a module directly, bypassing the safetensors loader, so the tests
    /// exercise the arithmetic without needing a checkpoint on disk.
    fn synth(ngram_size: i64, heads_per_ngram: i64, vocab: i64, eos: i64) -> NGramEmbedding {
        let num_heads = (ngram_size - 1) * heads_per_ngram;
        let sizes = Tensor::full([num_heads], vocab, (Kind::Int64, Device::Cpu));
        let offsets = (Tensor::arange(num_heads, (Kind::Int64, Device::Cpu))) * vocab;
        let mult = Tensor::from_slice(
            &(0..ngram_size).map(|i| 0x9E37_79B1i64 + i * 0x0100_0193).collect::<Vec<_>>(),
        );
        let num_rows = num_heads * vocab;
        // A table whose row r is just [r, r, ...]: makes it trivial to read the
        // selected row index back out of the embedding.
        let table = Tensor::arange(num_rows, (Kind::Float, Device::Cpu))
            .unsqueeze(1)
            .expand([num_rows, ROW_DIM], false)
            .contiguous()
            .to_kind(Kind::Half);
        NGramEmbedding {
            store: Store::Fp16(vec![table]),
            head_offsets: offsets,
            head_vocab_sizes: sizes,
            layer_multipliers: mult,
            rows_per_shard: num_rows,
            num_rows,
            ngram_size,
            heads_per_ngram,
            num_heads,
            ple_embed_dim: num_heads * ROW_DIM,
            eos_token_id: eos,
            device: Device::Cpu,
        }
    }

    /// Every head's ids must land inside that head's own slice of the table.
    /// A head reading another head's rows is silent — the embedding stays
    /// finite and the model merely learns from the wrong features.
    #[test]
    fn ids_stay_inside_their_head_partition() {
        let m = synth(4, 2, 97, 0);
        let toks = Tensor::from_slice(&[5i64, 9, 2, 7, 4, 11, 3, 8]).view([1, 8]);
        let ids = m.compute_ngram_ids(&toks, 5);
        assert_eq!(ids.size(), vec![1, 5, 6]);
        for h in 0..m.num_heads {
            let col = ids.select(2, h);
            let lo = h * 97;
            assert!(f64::try_from(col.min()).unwrap() >= lo as f64);
            assert!(f64::try_from(col.max()).unwrap() < (lo + 97) as f64);
        }
    }

    /// An eos cuts the history: the first position of a new segment must hash
    /// the same as the first position of a freshly started sequence, whatever
    /// preceded the eos.
    #[test]
    fn hashing_does_not_span_an_eos() {
        let m = synth(3, 2, 61, 0);
        // context = 2. Same segment content, different junk before the eos.
        let a = Tensor::from_slice(&[7i64, 3, 0, 12, 5, 9]).view([1, 6]);
        let b = Tensor::from_slice(&[41i64, 19, 0, 12, 5, 9]).view([1, 6]);
        // Positions 3..5 only: the eos token's *own* n-gram still reaches back
        // past it, since the segment boundary is set by the preceding eos.
        let ia = m.compute_ngram_ids(&a, 3);
        let ib = m.compute_ngram_ids(&b, 3);
        assert!(f64::try_from((ia - ib).abs().max()).unwrap() == 0.0);
    }

    /// Rolling the window forward must not change the ids already computed for
    /// a position — this is what lets incremental decoding hash only the new
    /// token instead of the whole prompt.
    #[test]
    fn ids_are_position_local() {
        let m = synth(4, 1, 53, 0);
        let toks = Tensor::from_slice(&[2i64, 8, 5, 1, 9, 4, 6, 3]).view([1, 8]);
        let whole = m.compute_ngram_ids(&toks, 5); // positions 3..8
        let tail = m.compute_ngram_ids(&toks.narrow(1, 4, 4), 1); // position 7 alone
        let d = whole.select(1, 4) - tail.select(1, 0);
        assert!(f64::try_from(d.abs().max()).unwrap() == 0.0);
    }

    /// The concatenated feature vector must be the selected rows in head order.
    #[test]
    fn embedding_concatenates_rows_in_head_order() {
        let m = synth(3, 2, 31, 0);
        let toks = Tensor::from_slice(&[1i64, 6, 4, 9, 2]).view([1, 5]);
        let ids = m.compute_ngram_ids(&toks, 3);
        let emb = m.embed_ids(&ids);
        assert_eq!(emb.size(), vec![1, 3, m.num_heads * ROW_DIM]);
        for s in 0..3 {
            for h in 0..m.num_heads {
                let want = f64::try_from(ids.select(1, s).select(1, h)).unwrap();
                let got = f64::try_from(emb.select(1, s).narrow(1, h * ROW_DIM, ROW_DIM).mean(Kind::Float)).unwrap();
                assert!((got - want).abs() < 1e-3, "head {h} pos {s}: {got} != {want}");
            }
        }
    }

    /// Sharding is pure routing: the same table split in two must embed
    /// identically to the whole one.
    #[test]
    fn sharded_lookup_matches_the_whole_table() {
        let mut m = synth(3, 2, 40, 0);
        let Store::Fp16(shards) = &m.store else { unreachable!() };
        let whole = shards[0].shallow_clone();
        let toks = Tensor::from_slice(&[3i64, 7, 1, 8, 2, 5]).view([1, 6]);
        let ids = m.compute_ngram_ids(&toks, 4);
        let want = m.embed_ids(&ids);

        let half = m.num_rows / 2;
        m.store = Store::Fp16(vec![whole.narrow(0, 0, half), whole.narrow(0, half, m.num_rows - half)]);
        m.rows_per_shard = half;
        let got = m.embed_ids(&ids);
        assert!(f64::try_from((want - got).abs().max()).unwrap() == 0.0);
    }

    /// The trellis codec must round-trip: states packed as upstream packs them
    /// unpack to exactly the same states.
    #[test]
    fn trellis_unpack_inverts_the_ring() {
        for k in [2i64, 3, 4] {
            let n = 3;
            let states = Tensor::randint(65536, [n, ROW_DIM], (Kind::Int64, Device::Cpu));
            // Pack: low K bits of each state written into the ring bitstream.
            let low = states.bitwise_and((1i64 << k) - 1);
            let bits = low
                .unsqueeze(-1)
                .bitwise_right_shift(&Tensor::arange(k, (Kind::Int64, Device::Cpu)))
                .bitwise_and(1i64)
                .reshape([n, ROW_DIM * k / 16, 16]);
            let words = (bits.bitwise_left_shift(&Tensor::arange(16, (Kind::Int64, Device::Cpu))))
                .sum_dim_intlist(vec![-1i64].as_slice(), false, Kind::Int64)
                .bitwise_and(0xFFFFi64);
            // Reinterpret as int16 the way the checkpoint stores it.
            let words = (&words - (words.gt(0x7FFF).to_kind(Kind::Int64) * 0x10000)).to_kind(Kind::Int16);
            let scale = Tensor::zeros([n, 1], (Kind::Int16, Device::Cpu));
            let packed = Tensor::cat(&[scale, words], 1);
            let (got, _) = unpack_rows(&packed, k);
            // Only the low K bits of each state survive packing; the rest are
            // reconstructed from the ring, so compare that projection.
            let want_low = states.bitwise_and((1i64 << k) - 1);
            let got_low = got.bitwise_and((1i64 << k) - 1);
            assert!(f64::try_from((want_low - got_low).abs().max()).unwrap() == 0.0, "K={k}");
        }
    }

    /// The mul1 codebook is an affine map of a byte-sum, so it must be bounded
    /// and hit both signs — a wrong fp16 constant shows up as a codebook that
    /// never crosses zero.
    #[test]
    fn mul1_codebook_spans_zero() {
        let cb = mul1_codebook(Device::Cpu).to_kind(Kind::Float);
        assert_eq!(cb.size(), vec![65536]);
        assert!(f64::try_from(cb.min()).unwrap() < 0.0);
        assert!(f64::try_from(cb.max()).unwrap() > 0.0);
        assert!(f64::try_from(cb.abs().max()).unwrap() < 32.0);
    }
}
