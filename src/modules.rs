//! Ports of `modules/{linear,quant/exl3,quant/fp16,rmsnorm,embedding,attn,mlp,transformer}.py`.
//! Grades per PLAN.md. Qwen3 single-device, no-cache inference path only.

use crate::config::Config;
use crate::ffi;
use crate::rope::RoPE;
use crate::safetensors::SafeTensors;
use anyhow::{bail, Result};
use std::sync::OnceLock;
use tch::{Device, Kind, Tensor};

/// `EXL3_ATTN_PROF=1`: accumulate per-call quant-KV dequant / attention time so
/// the generator can print a per-decode-step breakdown. Debug only — the timers
/// synchronize the stream.
static ATTN_PROF: OnceLock<bool> = OnceLock::new();
/// `EXL3_KVQ_WINDOW=1`: force the legacy bulk-dequant fp16 window path.
static KVQ_WINDOW: OnceLock<bool> = OnceLock::new();
pub static DEQ_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ATTN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
use std::sync::atomic::Ordering;

fn pad128(x: i64) -> i64 {
    (x + 127) / 128 * 128
}

/// How attention should source its K/V this call.
pub enum Attn<'a> {
    /// Single self-contained pass over this call's own K/V (`forward_last`).
    /// `past_len` only shifts the RoPE phase.
    NoCache { past_len: i64 },
    /// Paged KV cache: append this call's K/V, attend over the stored prefix.
    /// `seqlens` (int32 `[1]`, device) is the pre-append length and doubles as
    /// the RoPE `positions` vector, so a captured graph replays at any offset.
    /// `rope_table`, when set, is a `[max_pos, rotary_dim/2]` f32 angle table for
    /// MRoPE (multimodal sequences) — see [`crate::rope::RoPE::apply`].
    Paged {
        k_cache: &'a Tensor,
        v_cache: &'a Tensor,
        block_table: &'a Tensor,
        seqlens: &'a Tensor,
        rope_table: Option<&'a Tensor>,
    },
    /// Quantized paged KV cache: quantize this call's K/V into `qk/qv/sk/sv` at
    /// `seqlens`, dequantize the stored prefix into the shared fp16 `k_scratch`/
    /// `v_scratch` pools, then attend as `Paged`.
    PagedQuant {
        qk: &'a Tensor,
        qv: &'a Tensor,
        sk: &'a Tensor,
        sv: &'a Tensor,
        /// fp16 dequant scratch. `Some` = a caller-owned full-pool pair (the
        /// single-seq / CUDA-graph path). `None` = allocate a compact
        /// `bsz*pages_per_seq` scratch per call and free it (the batched path —
        /// no fixed pool-sized fp16 cost).
        k_scratch: Option<&'a Tensor>,
        v_scratch: Option<&'a Tensor>,
        block_table: &'a Tensor,
        seqlens: &'a Tensor,
        compand_a: f32,
        /// MRoPE angle table, as in [`Attn::Paged`] (applied to the fresh K/V
        /// before they are quantized into the store).
        rope_table: Option<&'a Tensor>,
    },
}

// ---------------------------------------------------------------------------
// Linear (fp16 + EXL3)  — modules/linear.py + quant/{exl3,fp16}.py (grade B)
// ---------------------------------------------------------------------------

enum Inner {
    Fp16 {
        weight: Tensor, // (in, out)
        bias: Option<Tensor>,
    },
    Exl3 {
        trellis: Tensor, // (k/16, n/16, 16K) int16
        suh: Tensor,     // (in) f16
        svh: Tensor,     // (out) f16
        k: i64,
        mcg: bool,
        mul1: bool,
        bias: Option<Tensor>,
    },
}

pub struct Linear {
    pub in_features: i64,
    pub out_features: i64,
    in_unpadded: i64,
    out_unpadded: i64,
    trim_padded_out: bool,
    softcap: f64,
    inner: Inner,
}

impl Linear {
    pub fn load(
        stc: &SafeTensors,
        key: &str,
        alt_key: Option<&str>,
        in_features: i64,
        out_features: i64,
        device: Device,
        trim_padded_out: bool,
        softcap: f64,
    ) -> Result<Self> {
        let inf = pad128(in_features);
        let outf = pad128(out_features);
        for k in [Some(key), alt_key].into_iter().flatten() {
            if stc.has_group(k, &["trellis"]) {
                let trellis = stc.get(&format!("{k}.trellis"), device, false, false)?;
                let kk = trellis.size()[2] / 16;
                let suh = match (stc.has(&format!("{k}.suh")), stc.has(&format!("{k}.su"))) {
                    (true, _) => stc.get(&format!("{k}.suh"), device, false, false)?,
                    _ => bail!("{k}: only unpacked suh/svh supported in this port"),
                };
                let svh = stc.get(&format!("{k}.svh"), device, false, false)?;
                let bias = stc.get_opt(&format!("{k}.bias"), device);
                return Ok(Self {
                    in_features: inf,
                    out_features: outf,
                    in_unpadded: in_features,
                    out_unpadded: out_features,
                    trim_padded_out,
                    softcap,
                    inner: Inner::Exl3 {
                        trellis,
                        suh,
                        svh,
                        k: kk,
                        mcg: stc.has(&format!("{k}.mcg")),
                        mul1: stc.has(&format!("{k}.mul1")),
                        bias,
                    },
                });
            }
            if stc.has_group(k, &["weight"]) {
                // (out,in) checkpoint -> transpose to (in,out) like transposed_load
                let mut weight = stc
                    .get(&format!("{k}.weight"), device, true, false)?
                    .transpose(0, 1)
                    .contiguous();
                if weight.size() != [inf, outf] {
                    let pad = Tensor::zeros([inf, outf], (Kind::Half, device));
                    pad.narrow(0, 0, weight.size()[0])
                        .narrow(1, 0, weight.size()[1])
                        .copy_(&weight);
                    weight = pad;
                }
                let bias = stc
                    .get_opt(&format!("{k}.bias"), device)
                    .map(|b| b.to_kind(Kind::Half));
                return Ok(Self {
                    in_features: inf,
                    out_features: outf,
                    in_unpadded: in_features,
                    out_unpadded: out_features,
                    trim_padded_out,
                    softcap,
                    inner: Inner::Fp16 { weight, bias },
                });
            }
        }
        bail!("No supported quant tensors for {key}");
    }

    /// The EXL3 weight tensors and codebook flags, or `None` when this layer is
    /// plain fp16. The fused MoE kernel takes raw per-expert data pointers, so it
    /// needs to reach inside; nothing else does.
    pub fn exl3_parts(&self) -> Option<(&Tensor, &Tensor, &Tensor, i64, bool, bool)> {
        match &self.inner {
            Inner::Exl3 { trellis, suh, svh, k, mcg, mul1, bias } => {
                // A bias would have to be added after the fused kernel's own
                // accumulation; no MoE checkpoint in the wild carries one, so
                // rather than guess, decline the fast path.
                bias.is_none().then_some((trellis, suh, svh, *k, *mcg, *mul1))
            }
            Inner::Fp16 { .. } => None,
        }
    }

    /// `Linear.forward`: zero-extend narrow input, run inner, trim padded output, softcap.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let last = *x.size().last().unwrap();
        let x = if last < self.in_features {
            x.pad([0, self.in_features - last], "constant", 0.0)
        } else {
            x.shallow_clone()
        };
        let mut shape = x.size();
        let rows: i64 = shape[..shape.len() - 1].iter().product();
        // decode inputs (norm / activation outputs) are already contiguous fp16,
        // so skip the redundant dtype copy — this runs 7×/layer/token.
        let x2r = x.reshape([rows, self.in_features]);
        let x2 = if x2r.kind() == Kind::Half {
            x2r.contiguous()
        } else {
            x2r.to_kind(Kind::Half).contiguous()
        };

        let mut y = match &self.inner {
            Inner::Fp16 { weight, bias } => {
                let y = Tensor::empty([rows, self.out_features], (Kind::Half, x2.device()));
                ffi::hgemm(&x2, weight, &y);
                match bias {
                    Some(b) => y + b,
                    None => y,
                }
            }
            Inner::Exl3 {
                trellis,
                suh,
                svh,
                k,
                mcg,
                mul1,
                bias,
            } => {
                // Prefill (rows > 144): dequantize the weight to dense fp16 once
                // and run a real tensor-core GEMM — the trellis GEMM re-derives
                // the weight per m-tile and gets ~no batching benefit. Mirrors
                // `quant/exl3.py::reconstruct_hgemm` (non-fused path). lm_head's
                // huge N (> 32768) keeps the trellis path (it's always m == 1).
                let y = Tensor::empty([rows, self.out_features], (Kind::Half, x2.device()));
                if rows > 144 && self.out_features <= 32768 && self.out_features == self.out_unpadded {
                    let xh = Tensor::empty_like(&x2);
                    ffi::had_r_128(&x2, &xh, Some(suh), None, 1.0);
                    let w = Tensor::empty([self.in_features, self.out_features], (Kind::Half, x2.device()));
                    ffi::reconstruct(&w, trellis, *k, *mcg, *mul1);
                    ffi::hgemm(&xh, &w, &y);
                    ffi::had_r_128(&y, &y, None, Some(svh), 1.0);
                } else {
                    let a_had = Tensor::empty_like(&x2);
                    ffi::exl3_gemm(&x2, trellis, &y, suh, &a_had, svh, *mcg, *mul1);
                }
                match bias {
                    Some(b) => y + b,
                    None => y,
                }
            }
        };

        *shape.last_mut().unwrap() = self.out_features;
        y = y.reshape(&shape[..]);
        if self.trim_padded_out && self.out_features != self.out_unpadded {
            y = y.narrow(shape.len() as i64 - 1, 0, self.out_unpadded).contiguous();
        }
        if self.softcap != 0.0 {
            y = (y / self.softcap).tanh() * self.softcap;
        }
        y
    }

    /// Device bytes held by this layer's weights.
    pub fn nbytes(&self) -> i64 {
        let t = |t: &Tensor| t.numel() as i64 * t.kind().elt_size_in_bytes() as i64;
        match &self.inner {
            Inner::Fp16 { weight, bias } => t(weight) + bias.as_ref().map_or(0, t),
            Inner::Exl3 { trellis, suh, svh, bias, .. } => {
                t(trellis) + t(suh) + t(svh) + bias.as_ref().map_or(0, t)
            }
        }
    }

    /// A copy of this (EXL3-quantised) layer restricted to a subset of its
    /// output features, given as `[start, end)` ranges.
    ///
    /// Exact on the features it keeps. The trellis is tiled `(k/16, n/16, 16K)`
    /// over the output dim and the output rotation `svh` is applied by
    /// `had_r_128` in independent 128-wide blocks, so a 128-aligned slice of the
    /// output is computed byte-for-byte as it would be in the full layer — this
    /// drops work, it does not approximate it.
    ///
    /// The point is `lm_head`: at 4 bits it is 1.27 GB, a decode step is
    /// bandwidth-bound, and a speculative *draft* pass only needs the argmax.
    /// Dropping the rare tail of the vocabulary makes the draft head several
    /// times cheaper, and because verification is exact, a token the pruned head
    /// can no longer propose is simply not drafted — it costs acceptance, never
    /// correctness. Returns the layer and the kept output ids, so the caller can
    /// map a compact argmax back to a real token id.
    pub fn prune_out(&self, ranges: &[(i64, i64)]) -> Result<(Self, Vec<i64>)> {
        const BLK: i64 = 128;
        let Inner::Exl3 { trellis, suh, svh, k, mcg, mul1, bias } = &self.inner else {
            bail!("prune_out: only supported for EXL3-quantised layers");
        };
        if bias.is_some() {
            bail!("prune_out: biased layers not supported");
        }
        // Ranges may overlap and arrive unsorted (a frequency prefix plus a
        // scatter of pinned ids), so reduce them to a sorted set of 128-blocks
        // first; that also keeps the kept-id list ascending, which is what makes
        // the compact index -> token id map well defined.
        let blocks: std::collections::BTreeSet<i64> = ranges
            .iter()
            .flat_map(|&(a, b)| {
                let a = (a / BLK).max(0);
                let b = ((b + BLK - 1) / BLK).min(self.out_features / BLK);
                a..b
            })
            .collect();
        let mut ids: Vec<i64> = Vec::with_capacity(blocks.len() * BLK as usize);
        let mut tiles: Vec<i64> = Vec::with_capacity(blocks.len() * (BLK / 16) as usize);
        for blk in blocks {
            ids.extend((blk * BLK)..(blk * BLK + BLK));
            tiles.extend((blk * BLK / 16)..((blk * BLK + BLK) / 16));
        }
        if ids.is_empty() {
            bail!("prune_out: empty keep set");
        }
        let dev = trellis.device();
        let idx = Tensor::from_slice(&tiles).to_device(dev);
        let trellis = trellis.index_select(1, &idx).contiguous();
        let svh = svh.index_select(0, &Tensor::from_slice(&ids).to_device(dev)).contiguous();
        let out = ids.len() as i64;
        Ok((
            Self {
                in_features: self.in_features,
                out_features: out,
                in_unpadded: self.in_unpadded,
                out_unpadded: out,
                trim_padded_out: false,
                softcap: self.softcap,
                inner: Inner::Exl3 {
                    trellis,
                    suh: suh.shallow_clone(),
                    svh,
                    k: *k,
                    mcg: *mcg,
                    mul1: *mul1,
                    bias: None,
                },
            },
            ids,
        ))
    }
}

// ---------------------------------------------------------------------------
// RMSNorm — modules/rmsnorm.py (grade A)
// ---------------------------------------------------------------------------

pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
    constant_bias: f32,
    constant_scale: f32,
}

impl RmsNorm {
    pub fn load(stc: &SafeTensors, key: &str, eps: f32, device: Device) -> Result<Self> {
        Self::load_biased(stc, key, eps, 0.0, device)
    }

    /// As `load`, but with a `constant_bias` added to the norm weight (Qwen3.5
    /// uses `1.0` on every RMSNorm).
    pub fn load_biased(
        stc: &SafeTensors,
        key: &str,
        eps: f32,
        constant_bias: f32,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: stc.get(&format!("{key}.weight"), device, true, false)?,
            eps,
            constant_bias,
            constant_scale: 1.0,
        })
    }

    /// out dtype = half (matches every call site in the Qwen3 tree).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.size();
        let dim = *shape.last().unwrap();
        let x2 = x.reshape([-1, dim]).contiguous();
        let y2 = Tensor::empty_like(&x2).to_kind(Kind::Half);
        ffi::rms_norm(&x2, &self.weight, &y2, self.eps, self.constant_bias, self.constant_scale);
        y2.reshape(&shape[..])
    }

    /// Pre-norm residual add fused with the norm (`norm.cu` `RES_IN`): folds
    /// `add` (a preceding sublayer's fp16 output) into the fp32 residual `r`
    /// **in place** (`r += add`, rounded exactly as an unfused add would be),
    /// then returns `norm(r)`. This replaces a standalone `x + y` elementwise
    /// kernel per sublayer. `add == None` ⇒ plain [`forward`] on `r`.
    ///
    /// Requires a contiguous `r` (the in-place write goes through a reshape
    /// view); falls back to the unfused path otherwise.
    pub fn forward_res(&self, r: &Tensor, add: Option<&Tensor>) -> Tensor {
        let Some(add) = add else {
            return self.forward(r);
        };
        if !r.is_contiguous() {
            return self.forward(&(r + add));
        }
        let shape = r.size();
        let dim = *shape.last().unwrap();
        let r2 = r.reshape([-1, dim]);
        let add2 = add.reshape([-1, dim]).contiguous();
        let y2 = Tensor::empty([r2.size()[0], dim], (Kind::Half, r.device()));
        ffi::rms_norm_res_in(
            &add2,
            Some(&self.weight),
            &y2,
            &r2,
            self.eps,
            self.constant_bias,
            self.constant_scale,
        );
        y2.reshape(&shape[..])
    }
}

// ---------------------------------------------------------------------------
// Embedding — modules/embedding.py (grade B, plain path)
// ---------------------------------------------------------------------------

pub struct Embedding {
    weight: Tensor, // (vocab, hidden) f16, on device
    out_kind: Kind,
}

impl Embedding {
    pub fn load(stc: &SafeTensors, key: &str, device: Device) -> Result<Self> {
        Ok(Self {
            weight: stc.get(&format!("{key}.weight"), device, true, true)?,
            out_kind: Kind::Float,
        })
    }
    pub fn forward(&self, ids: &Tensor) -> Tensor {
        // ids: (1, seq) int64
        let flat = ids.reshape([-1]);
        let emb = self.weight.index_select(0, &flat);
        emb.reshape([ids.size()[0], ids.size()[1], self.weight.size()[1]])
            .to_kind(self.out_kind)
    }
}

// ---------------------------------------------------------------------------
// Attention — modules/attn.py decode_flash_attn_nc (grade C)
// ---------------------------------------------------------------------------

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Per-head Q/K RMSNorm weights, fused into the RoPE kernel. `None` on
    /// Llama-shaped and GLM4 stacks, which carry no `q_norm`/`k_norm` tensors —
    /// the kernel then skips the norm rather than applying an identity one.
    q_norm_w: Option<Tensor>,
    k_norm_w: Option<Tensor>,
    norm_eps: f32,
    rope: RoPE,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    sm_scale: f64,
    /// Qwen3.5 full-attention: interleaved `[q | gate]` q_proj, `o *= sigmoid(gate)`.
    output_gate: bool,
    /// `constant_bias` for the fused Q/K RMSNorm inside RoPE (1.0 on Qwen3.5).
    norm_cbias: f32,
    /// KV-head repeat factor to reach a kernel-supported GQA ratio (1 = native).
    kv_repeat: i64,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn load(stc: &SafeTensors, key: &str, cfg: &Config, device: Device) -> Result<Self> {
        let hd = cfg.head_dim;
        let gate = cfg.attn_output_gate;
        let q_out = if gate { cfg.num_q_heads * hd * 2 } else { cfg.num_q_heads * hd };
        Ok(Self {
            q_proj: Linear::load(stc, &format!("{key}.q_proj"), None, cfg.hidden_size, q_out, device, true, 0.0)?,
            k_proj: Linear::load(stc, &format!("{key}.k_proj"), None, cfg.hidden_size, cfg.num_kv_heads * hd, device, true, 0.0)?,
            v_proj: Linear::load(stc, &format!("{key}.v_proj"), None, cfg.hidden_size, cfg.num_kv_heads * hd, device, true, 0.0)?,
            o_proj: Linear::load(stc, &format!("{key}.o_proj"), None, cfg.num_q_heads * hd, cfg.hidden_size, device, true, 0.0)?,
            q_norm_w: cfg
                .qk_norm
                .then(|| stc.get(&format!("{key}.q_norm.weight"), device, true, false))
                .transpose()?,
            k_norm_w: cfg
                .qk_norm
                .then(|| stc.get(&format!("{key}.k_norm.weight"), device, true, false))
                .transpose()?,
            norm_eps: cfg.rms_norm_eps,
            rope: RoPE::new(device, &cfg.rope),
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: hd,
            sm_scale: (hd as f64).powf(-0.5),
            output_gate: gate,
            norm_cbias: cfg.norm_constant_bias,
            kv_repeat: cfg.kv_heads_eff().1,
        })
    }

    /// x: (1, seq, hidden) half.
    /// Project a hidden stream straight into normed, RoPE'd K/V — no query, no
    /// attention. This is DFlash2's `update_kv_from_target`: the draft model's
    /// context K/V are derived from the *target's* hidden states rather than by
    /// running the drafter over the context, which is what makes its context
    /// effectively free.
    ///
    /// `x` `[b, l, H]`, `positions` int32 `[b]` giving each row's base absolute
    /// position (RoPE advances across `l` from there). Returns `(k, v)`, each
    /// `[b, l, kv_heads, head_dim]` fp16, ready to store.
    pub fn kv_from_hidden(&self, x: &Tensor, positions: &Tensor) -> (Tensor, Tensor) {
        let (b, s) = (x.size()[0], x.size()[1]);
        let k = self
            .k_proj
            .forward(x)
            .reshape([b, s, self.num_kv_heads, self.head_dim])
            .contiguous();
        let v = self
            .v_proj
            .forward(x)
            .reshape([b, s, self.num_kv_heads, self.head_dim])
            .contiguous()
            .to_kind(Kind::Half);
        // K rides alone here — `apply` would also treat it as its own companion
        // K and rotate it twice with a null norm weight.
        self.rope.apply_one(
            &k,
            0,
            Some(positions),
            self.k_norm_w.as_ref(),
            self.norm_eps,
            self.norm_cbias,
        );
        (k, v)
    }

    /// Attention for a DFlash2 draft block: the `s` block rows attend to each
    /// other **bidirectionally** and to a sliding left window of pre-computed
    /// context K/V.
    ///
    /// Done in tch rather than through the paged kernels on purpose. Those take
    /// a scalar `causal_limit` and have no mask input, so bidirectional rows are
    /// not expressible there; but the block is only `s = 8` rows and the window
    /// caps the context, so the whole thing is a `[s, window + s]` score matrix
    /// — small enough that the naive form costs nothing measurable and is exact.
    ///
    /// `x` `[b, s, H]` post-norm block input; `ctx_k`/`ctx_v`
    /// `[b, L, kv_heads, head_dim]` already normed and RoPE'd (see
    /// [`Attention::kv_from_hidden`]) with `ctx_k[..0..]` at absolute position
    /// `ctx_start`; block row `i` sits at absolute position `base_pos + i`.
    /// `window <= 0` means unlimited.
    pub fn forward_block_windowed(
        &self,
        x: &Tensor,
        ctx_k: &Tensor,
        ctx_v: &Tensor,
        ctx_start: i64,
        base_pos: i64,
        window: i64,
    ) -> Tensor {
        let (b, s) = (x.size()[0], x.size()[1]);
        let dev = x.device();
        let q = self
            .q_proj
            .forward(x)
            .reshape([b, s, self.num_q_heads, self.head_dim])
            .contiguous();
        let k = self
            .k_proj
            .forward(x)
            .reshape([b, s, self.num_kv_heads, self.head_dim])
            .contiguous();
        let v = self
            .v_proj
            .forward(x)
            .reshape([b, s, self.num_kv_heads, self.head_dim])
            .contiguous()
            .to_kind(Kind::Half);
        // Contiguous block positions, so the scalar RoPE offset is enough.
        self.rope.apply(
            &q,
            &k,
            base_pos,
            None,
            self.q_norm_w.as_ref(),
            self.k_norm_w.as_ref(),
            self.norm_eps,
            self.norm_cbias,
            None,
        );

        let l = ctx_k.size()[1];
        let kk = Tensor::cat(&[ctx_k.shallow_clone(), k], 1); // [b, L+s, kvh, hd]
        let vv = Tensor::cat(&[ctx_v.shallow_clone(), v], 1);
        // GQA: expand KV heads up to the query heads
        let rep = self.num_q_heads / self.num_kv_heads;
        let (kk, vv) = if rep > 1 {
            (
                kk.repeat_interleave_self_int(rep, 2, None),
                vv.repeat_interleave_self_int(rep, 2, None),
            )
        } else {
            (kk, vv)
        };
        // [b, heads, s, hd] x [b, heads, hd, L+s] -> [b, heads, s, L+s]
        let qh = q.permute([0, 2, 1, 3]).to_kind(Kind::Float);
        let kh = kk.permute([0, 2, 3, 1]).to_kind(Kind::Float);
        let vh = vv.permute([0, 2, 1, 3]).to_kind(Kind::Float).contiguous();
        let mut scores = qh.matmul(&kh) * self.sm_scale;

        // Key absolute positions: context `ctx_start..ctx_start+L`, then the
        // block itself at `base_pos..base_pos+s`.
        let key_pos = Tensor::cat(
            &[
                Tensor::arange_start(ctx_start, ctx_start + l, (Kind::Float, dev)),
                Tensor::arange_start(base_pos, base_pos + s, (Kind::Float, dev)),
            ],
            0,
        )
        .reshape([1, 1, 1, l + s]);
        let q_pos = Tensor::arange_start(base_pos, base_pos + s, (Kind::Float, dev)).reshape([1, 1, s, 1]);
        if window > 0 {
            // drop context older than the window; block keys are never dropped
            // (they sit at or after `base_pos`, so their distance is <= s)
            let too_old = (&q_pos - &key_pos).ge(window as f64);
            scores = scores.masked_fill(&too_old, f64::NEG_INFINITY);
        }
        let out = scores
            .softmax(-1, Kind::Float)
            .matmul(&vh)
            .permute([0, 2, 1, 3])
            .reshape([b, s, self.num_q_heads * self.head_dim])
            .to_kind(Kind::Half);
        self.o_proj.forward(&out)
    }

    pub fn forward(&self, x: &Tensor, ctx: &Attn) -> Tensor {
        let (b, s) = (x.size()[0], x.size()[1]);
        let qraw = self.q_proj.forward(x);
        let (q, gate) = if self.output_gate {
            // interleaved `[q_head | gate_head]` per head → split
            let qg = qraw
                .reshape([b, s, self.num_q_heads, self.head_dim * 2])
                .to_kind(Kind::Half)
                .contiguous();
            let q = Tensor::empty([b, s, self.num_q_heads, self.head_dim], (Kind::Half, x.device()));
            let g = Tensor::empty([b, s, self.num_q_heads * self.head_dim], (Kind::Half, x.device()));
            ffi::deinterleave_qg(&qg, &q, &g, self.head_dim);
            (q, Some(g))
        } else {
            (
                qraw.reshape([b, s, self.num_q_heads, self.head_dim]).contiguous(),
                None,
            )
        };
        let k = self
            .k_proj
            .forward(x)
            .reshape([b, s, self.num_kv_heads, self.head_dim])
            .contiguous();
        let v = self
            .v_proj
            .forward(x)
            .reshape([b, s, self.num_kv_heads, self.head_dim])
            .contiguous()
            .to_kind(Kind::Half);

        // fused Q/K RMSNorm + RoPE, in place.
        let (past_len, positions, rope_table) = match ctx {
            Attn::NoCache { past_len } => (*past_len, None, None),
            Attn::Paged { seqlens, rope_table, .. } => (0, Some(*seqlens), *rope_table),
            Attn::PagedQuant { seqlens, rope_table, .. } => (0, Some(*seqlens), *rope_table),
        };
        self.rope.apply(
            &q,
            &k,
            past_len,
            positions,
            self.q_norm_w.as_ref(),
            self.k_norm_w.as_ref(),
            self.norm_eps,
            self.norm_cbias,
            rope_table,
        );

        // Repeat KV heads up to a kernel-supported GQA ratio if needed
        // (Qwen3.5: 4 KV heads → 12, ratio 24/12 = 2). Rotation is per-position
        // so repeating after RoPE is identical to repeating before.
        let (k, v) = if self.kv_repeat > 1 {
            (
                k.repeat_interleave_self_int(self.kv_repeat, 2, None).contiguous(),
                v.repeat_interleave_self_int(self.kv_repeat, 2, None).contiguous(),
            )
        } else {
            (k, v)
        };

        let o = Tensor::empty(
            [b, s, self.num_q_heads, self.head_dim],
            (Kind::Half, x.device()),
        );
        match ctx {
            Attn::NoCache { .. } => {
                ffi::bighead_attn(&q, &k, &v, &o, true, self.sm_scale as f32);
            }
            Attn::Paged { k_cache, v_cache, block_table, seqlens, .. } => {
                ffi::paged_kv_cache_update(&k, &v, k_cache, v_cache, block_table, seqlens);
                ffi::bighead_attn_paged(
                    &q, &k, &v, k_cache, v_cache, block_table, seqlens, &o,
                    self.sm_scale as f32,
                );
            }
            Attn::PagedQuant {
                qk, qv, sk, sv, k_scratch, v_scratch, block_table, seqlens, compand_a, ..
            } => match (k_scratch, v_scratch) {
                // caller-owned full-pool fp16 scratch (single-seq / CUDA-graph path):
                // quantize fresh rows, bulk-dequant the prefix, attend fp16.
                (Some(ks), Some(vs)) => {
                    ffi::quant_cache_paged(&k, qk, sk, &v, qv, sv, seqlens, block_table, s, *compand_a, true);
                    ffi::dequant_cache_paged(qk, sk, ks, qv, sv, vs, seqlens, block_table, -1, *compand_a);
                    ffi::bighead_attn_paged(
                        &q, &k, &v, ks, vs, block_table, seqlens, &o, self.sm_scale as f32,
                    );
                }
                // batched path: quantize fresh rows, then attend straight off the
                // packed codes (`bighead_attn_paged_q`) — the query-tiled kernel
                // dequantizes each KV tile once into shared memory and shares it
                // across the CTA's query warps, so nothing materializes an fp16
                // KV window. The old path (bulk `dequant_cache_paged_window` into
                // a compact fp16 scratch, then the fp16 kernel) is still there as
                // the fallback for shapes the online kernel has no instantiation
                // for, and can be forced with `EXL3_KVQ_WINDOW=1`; it costs an
                // extra ~3.3 GB of write+read per decode step at 50k context.
                _ => {
                    let groups = sk.size()[2];
                    let bits = qk.size()[2] / groups;
                    let force_window =
                        *KVQ_WINDOW.get_or_init(|| std::env::var("EXL3_KVQ_WINDOW").is_ok());
                    let online = !force_window
                        && (self.head_dim == 128 || self.head_dim == 256)
                        && matches!(bits, 4 | 6 | 8)
                        // q_len 1 still goes through the per-query decode kernel;
                        // there is no query tile to amortize a dequant over.
                        && s > 1;
                    if online {
                        let prof =
                            *ATTN_PROF.get_or_init(|| std::env::var("EXL3_ATTN_PROF").is_ok());
                        let t0 = std::time::Instant::now();
                        ffi::bighead_attn_paged_q(
                            &q, &k, &v, qk, sk, qv, sv, block_table, seqlens, &o,
                            self.sm_scale as f32, *compand_a,
                        );
                        if prof {
                            tch::Cuda::synchronize(0);
                            ATTN_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                    } else {
                        let prof = *ATTN_PROF.get_or_init(|| std::env::var("EXL3_ATTN_PROF").is_ok());
                        let t0 = std::time::Instant::now();
                        ffi::quant_cache_paged(&k, qk, sk, &v, qv, sv, seqlens, block_table, s, *compand_a, true);
                        let bt = block_table.size();
                        let (bsz, pps) = (bt[0], bt[1]);
                        let kh = k.size()[2];
                        let hd = k.size()[3];
                        let dev = q.device();
                        let ks = Tensor::empty([bsz * pps, 256, kh, hd], (Kind::Half, dev));
                        let vs = Tensor::empty([bsz * pps, 256, kh, hd], (Kind::Half, dev));
                        ffi::dequant_cache_paged_window(qk, sk, &ks, qv, sv, &vs, seqlens, block_table, 0, *compand_a);
                        if prof {
                            tch::Cuda::synchronize(0);
                            DEQ_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                        let t1 = std::time::Instant::now();
                        let ident = Tensor::arange(bsz * pps, (Kind::Int, dev)).reshape([bsz, pps]);
                        ffi::bighead_attn_paged(
                            &q, &k, &v, &ks, &vs, &ident, seqlens, &o, self.sm_scale as f32,
                        );
                        if prof {
                            tch::Cuda::synchronize(0);
                            ATTN_NS.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                    }
                }
            },
        }

        let o = o.reshape([b, s, self.num_q_heads * self.head_dim]).contiguous();
        if let Some(g) = &gate {
            ffi::mul_sigmoid_(&o, g);
        }
        self.o_proj.forward(&o)
    }

    /// Q/K/V projection with the fused Q/K norm + RoPE applied, before any
    /// attention. `[b, s, hidden]` in; `(q, k, v, gate)` out, each
    /// `[b, s, heads, head_dim]` fp16 (`gate` is `[b, s, q_heads * head_dim]`,
    /// `None` unless the layer carries an interleaved output gate).
    ///
    /// Split out of `forward` for the QSA path, which has to score the indexer
    /// and build a selection mask between the projection and the attention.
    pub fn project_qkv(&self, x: &Tensor, past_len: i64) -> (Tensor, Tensor, Tensor, Option<Tensor>) {
        let (b, s) = (x.size()[0], x.size()[1]);
        let qraw = self.q_proj.forward(x);
        let (q, gate) = if self.output_gate {
            let qg = qraw
                .reshape([b, s, self.num_q_heads, self.head_dim * 2])
                .to_kind(Kind::Half)
                .contiguous();
            let q = Tensor::empty([b, s, self.num_q_heads, self.head_dim], (Kind::Half, x.device()));
            let g = Tensor::empty([b, s, self.num_q_heads * self.head_dim], (Kind::Half, x.device()));
            ffi::deinterleave_qg(&qg, &q, &g, self.head_dim);
            (q, Some(g))
        } else {
            (qraw.reshape([b, s, self.num_q_heads, self.head_dim]).contiguous(), None)
        };
        let k = self
            .k_proj
            .forward(x)
            .reshape([b, s, self.num_kv_heads, self.head_dim])
            .contiguous();
        let v = self
            .v_proj
            .forward(x)
            .reshape([b, s, self.num_kv_heads, self.head_dim])
            .contiguous()
            .to_kind(Kind::Half);
        self.rope.apply(
            &q,
            &k,
            past_len,
            None,
            self.q_norm_w.as_ref(),
            self.k_norm_w.as_ref(),
            self.norm_eps,
            self.norm_cbias,
            None,
        );
        (q, k, v, gate)
    }

    /// Attention against a **contiguous** K/V cache under an explicit boolean
    /// mask, evaluated in tch rather than through the paged kernels.
    ///
    /// The kernels take a scalar causal limit and have no mask input, so a
    /// per-query sparse selection — which is exactly what QSA produces — is not
    /// expressible there; the same reasoning as
    /// [`Attention::forward_block_windowed`]. This materializes the
    /// `[q_heads, s, total]` score matrix, so it is correct at any length but
    /// costs dense attention's memory. Upstream's fast path instead gathers only
    /// the selected rows (see `src/qsa.rs`).
    ///
    /// `k_cache`/`v_cache` are `[1, max_len, kv_heads, head_dim]` fp16, written
    /// in place at `[past_len, past_len + s)`. `mask` is `[s, past_len + s]`
    /// bool, true = may attend, and must already include causality.
    pub fn forward_masked(
        &self,
        x: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        past_len: i64,
        mask: &Tensor,
    ) -> Tensor {
        let (b, s) = (x.size()[0], x.size()[1]);
        debug_assert_eq!(b, 1, "forward_masked is single-sequence");
        let (q, k, v, gate) = self.project_qkv(x, past_len);
        let _ = k_cache.narrow(1, past_len, s).copy_(&k.to_kind(k_cache.kind()));
        let _ = v_cache.narrow(1, past_len, s).copy_(&v);

        let total = past_len + s;
        // From the cache's own head count, so this stays correct whether or not
        // the caller sized it to a kernel-friendly GQA ratio.
        let rep = self.num_q_heads / k_cache.size()[2];
        // [heads, len, hd]
        let kk = k_cache
            .narrow(1, 0, total)
            .squeeze_dim(0)
            .repeat_interleave_self_int(rep, 1, None)
            .transpose(0, 1)
            .to_kind(Kind::Float);
        let vv = v_cache
            .narrow(1, 0, total)
            .squeeze_dim(0)
            .repeat_interleave_self_int(rep, 1, None)
            .transpose(0, 1)
            .to_kind(Kind::Float);
        let qq = q.squeeze_dim(0).transpose(0, 1).to_kind(Kind::Float);

        let scores = qq.matmul(&kk.transpose(-2, -1)) * self.sm_scale;
        let scores = scores.masked_fill(&mask.logical_not().unsqueeze(0), f64::NEG_INFINITY);
        let o = scores.softmax(-1, Kind::Float).matmul(&vv);

        let o = o
            .transpose(0, 1)
            .reshape([b, s, self.num_q_heads * self.head_dim])
            .to_kind(Kind::Half)
            .contiguous();
        if let Some(g) = &gate {
            ffi::mul_sigmoid_(&o, g);
        }
        self.o_proj.forward(&o)
    }
}

// ---------------------------------------------------------------------------
// GatedMLP — modules/mlp.py (grade B)
// ---------------------------------------------------------------------------

pub struct GatedMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl GatedMlp {
    pub fn load(stc: &SafeTensors, key: &str, cfg: &Config, device: Device) -> Result<Self> {
        Self::load_sized(stc, key, cfg.hidden_size, cfg.intermediate_size, device)
    }

    /// As `load`, but with the MLP width given explicitly. The shared expert of
    /// a MoE block is `moe_intermediate_size * n_shared_experts` wide rather
    /// than the config's dense `intermediate_size`.
    pub fn load_sized(
        stc: &SafeTensors,
        key: &str,
        hidden_size: i64,
        intermediate_size: i64,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            gate: Linear::load(stc, &format!("{key}.gate_proj"), None, hidden_size, intermediate_size, device, false, 0.0)?,
            up: Linear::load(stc, &format!("{key}.up_proj"), None, hidden_size, intermediate_size, device, false, 0.0)?,
            down: Linear::load(stc, &format!("{key}.down_proj"), None, intermediate_size, hidden_size, device, true, 0.0)?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let g = self.gate.forward(x);
        let u = self.up.forward(x);
        // fused `silu(g) * u -> a` (one kernel, graph-capturable) instead of
        // three tch ops. Matches upstream's `BC_GatedMLP` activation path.
        let a = Tensor::empty_like(&g);
        ffi::silu_mul(&g, &u, &a, 0.0);
        self.down.forward(&a)
    }
}

// ---------------------------------------------------------------------------
// TransformerBlock — modules/transformer.py (grade B)
// ---------------------------------------------------------------------------

pub struct TransformerBlock {
    attn_norm: RmsNorm,
    attn: Attention,
    /// GLM4 sandwich norm on the attention output, applied before the residual
    /// add. `None` on Llama/Qwen3, which add the sublayer output unmodified.
    attn_post_norm: Option<RmsNorm>,
    mlp_norm: RmsNorm,
    mlp: BlockMlp,
    /// GLM4 sandwich norm on the MLP output; see `attn_post_norm`.
    mlp_post_norm: Option<RmsNorm>,
}

/// A block's feed-forward: dense on most architectures, block-sparse experts on
/// the MoE ones. MoE checkpoints can mix the two — `decoder_sparse_step` and
/// `mlp_only_layers` leave some layers dense — so this is per layer, not per model.
pub enum BlockMlp {
    Dense(GatedMlp),
    Sparse(crate::moe::BlockSparseMlp),
}

impl BlockMlp {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            BlockMlp::Dense(m) => m.forward(x),
            BlockMlp::Sparse(m) => m.forward(x),
        }
    }
}

impl TransformerBlock {
    pub fn load(
        stc: &SafeTensors,
        key: &str,
        cfg: &Config,
        device: Device,
        layer_idx: i64,
    ) -> Result<Self> {
        let post = |stc: &SafeTensors, k: &str| -> Result<Option<RmsNorm>> {
            cfg.arch_kind
                .has_post_norms()
                .then(|| RmsNorm::load(stc, k, cfg.rms_norm_eps, device))
                .transpose()
        };
        Ok(Self {
            attn_norm: RmsNorm::load(stc, &format!("{key}.input_layernorm"), cfg.rms_norm_eps, device)?,
            attn: Attention::load(stc, &format!("{key}.self_attn"), cfg, device)?,
            attn_post_norm: post(stc, &format!("{key}.post_self_attn_layernorm"))?,
            mlp_norm: RmsNorm::load(stc, &format!("{key}.post_attention_layernorm"), cfg.rms_norm_eps, device)?,
            mlp: match &cfg.moe {
                Some(p) if p.is_sparse_layer(layer_idx) => BlockMlp::Sparse(
                    crate::moe::BlockSparseMlp::load(stc, &format!("{key}.mlp"), cfg, device)?,
                ),
                _ => BlockMlp::Dense(GatedMlp::load(stc, &format!("{key}.mlp"), cfg, device)?),
            },
            mlp_post_norm: post(stc, &format!("{key}.post_mlp_layernorm"))?,
        })
    }

    /// This block's block-sparse MLP, if it has one rather than a dense MLP.
    pub fn sparse_mlp(&self) -> Option<&crate::moe::BlockSparseMlp> {
        match &self.mlp {
            BlockMlp::Sparse(m) => Some(m),
            BlockMlp::Dense(_) => None,
        }
    }

    /// Fused-residual block forward. `resid` is the shared `(1, seq, hidden)`
    /// fp32 residual stream, **mutated in place**. `pending` is the previous
    /// sublayer's output whose residual add was deferred; it is folded into
    /// `resid` while computing this block's attention norm (one fused kernel
    /// instead of a separate `x + y`). Returns this block's MLP output — its
    /// own residual add is likewise deferred to the next consumer (the next
    /// block's `forward`, or `Model::head_res` / `finalize_head`).
    pub fn forward(&self, resid: &Tensor, ctx: &Attn, pending: Option<&Tensor>) -> Tensor {
        let a_in = self.attn_norm.forward_res(resid, pending);
        let a_out = self.attn.forward(&a_in, ctx);
        // The sandwich norm sits between the sublayer and the residual add, so
        // it composes with the deferral: norm here, let the next consumer add.
        let a_out = match &self.attn_post_norm {
            Some(n) => n.forward(&a_out),
            None => a_out,
        };
        let m_in = self.mlp_norm.forward_res(resid, Some(&a_out));
        let m_out = self.mlp.forward(&m_in);
        match &self.mlp_post_norm {
            Some(n) => n.forward(&m_out),
            None => m_out,
        }
    }
}
