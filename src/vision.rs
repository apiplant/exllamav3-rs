//! Qwen3-VL / Qwen3.5 vision tower — port of `architecture/qwen3_vl.py`
//! `Qwen3VLVisionModel` + `modules/arch_specific/qwen3_vl.py` (grade C).
//!
//! Dense bf16 weights (this checkpoint does not quantize the vision tower), run
//! once per image, so the whole tower is plain `tch` tensor math — no CUDA
//! kernels. Pipeline:
//!
//!   image → preprocess (smart-resize, normalize, patchify) → `[N, 1536]`
//!         → patch_embed (Linear 1536→1152) → + interpolated pos-embed
//!         → 27× { LayerNorm, 2D-RoPE MHSA (non-causal), LayerNorm, MLP+gelu }
//!         → merger (LayerNorm, 2×2 shuffle, Linear 4608→4608→gelu→5120)
//!         → `[N/4, 5120]` image embeddings, spliced into the text stream.

use crate::cache::Qwen35Cache;
use crate::config::{Config, VisionConfig};
use crate::model::Model;
use crate::safetensors::SafeTensors;
use crate::tokenizer::Tok;
use anyhow::{bail, Result};
use std::io::Write;
use std::time::Instant;
use tch::{Device, Kind, Tensor};

/// A loaded vision tower plus everything the text model needs to place its
/// output: the `[num_img_tokens, hidden]` embeddings and the 3-D MRoPE angle
/// table for the full prompt.
pub struct ImageEmbeds {
    /// `[n_tok, text_hidden]` f16, ready to splice between vision_start/end.
    pub embeddings: Tensor,
    /// Deepstack features, one `[n_tok, text_hidden]` per entry of
    /// `deepstack_visual_indexes`. Empty when the tower has no deepstack (which
    /// is the case for every Qwen3.5 checkpoint). Layer `i` of the *text* stack
    /// adds entry `i` at the image token positions, so the language model sees
    /// the image again at three different depths of the tower rather than only
    /// through the final merger.
    pub deepstack: Vec<Tensor>,
    pub grid_t: i64,
    pub grid_h: i64,
    pub grid_w: i64,
}

impl ImageEmbeds {
    pub fn num_tokens(&self) -> i64 {
        self.embeddings.size()[0]
    }
}

struct Lin {
    w_t: Tensor, // [in, out] (already transposed), f16
    b: Option<Tensor>,
}
impl Lin {
    fn load(stc: &SafeTensors, key: &str, dev: Device, bias: bool) -> Result<Self> {
        let w = stc.get(&format!("{key}.weight"), dev, true, false)?; // [out, in] f16
        let b = if bias {
            Some(stc.get(&format!("{key}.bias"), dev, true, false)?.to_kind(Kind::Float))
        } else {
            None
        };
        Ok(Self { w_t: w.transpose(0, 1).contiguous(), b })
    }
    fn fwd(&self, x: &Tensor) -> Tensor {
        // x: [..., in] -> [..., out], compute in f16, return f32
        let y = x.to_kind(Kind::Half).matmul(&self.w_t).to_kind(Kind::Float);
        match &self.b {
            Some(b) => y + b,
            None => y,
        }
    }
}

struct Ln {
    w: Tensor,
    b: Tensor,
    eps: f64,
}
impl Ln {
    fn load(stc: &SafeTensors, key: &str, dev: Device, eps: f64) -> Result<Self> {
        Ok(Self {
            w: stc.get(&format!("{key}.weight"), dev, false, false)?.to_kind(Kind::Float),
            b: stc.get(&format!("{key}.bias"), dev, false, false)?.to_kind(Kind::Float),
            eps,
        })
    }
    fn fwd(&self, x: &Tensor) -> Tensor {
        let d = *x.size().last().unwrap();
        x.f_layer_norm([d], Some(&self.w), Some(&self.b), self.eps, false)
            .unwrap()
    }
}

struct Block {
    norm1: Ln,
    norm2: Ln,
    qkv: Lin,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
}

/// `Qwen3VLVisionPatchMerger`: LayerNorm, a `merge_size**2` spatial shuffle, and
/// a two-layer MLP down to the text hidden size.
///
/// `postshuffle` swaps the first two: the deepstack mergers norm *after* the
/// shuffle (over the concatenated 2x2 group) while the final merger norms
/// before it (per patch). Same tensors, different grouping — and the difference
/// is invisible in the shapes, so it has to be carried explicitly.
struct Merger {
    norm: Ln,
    fc1: Lin,
    fc2: Lin,
    postshuffle: bool,
}

impl Merger {
    fn load(stc: &SafeTensors, key: &str, device: Device, postshuffle: bool) -> Result<Self> {
        Ok(Self {
            norm: Ln::load(stc, &format!("{key}.norm"), device, 1e-6)?,
            fc1: Lin::load(stc, &format!("{key}.linear_fc1"), device, true)?,
            fc2: Lin::load(stc, &format!("{key}.linear_fc2"), device, true)?,
            postshuffle,
        })
    }

    /// `x` `[n, hidden]` -> `[n / merge**2, out_hidden]`.
    fn fwd(&self, x: &Tensor, merge_sq: i64) -> Tensor {
        let y = match self.postshuffle {
            true => self.norm.fwd(&x.reshape([x.size()[0] / merge_sq, -1])),
            false => self.norm.fwd(x).reshape([x.size()[0] / merge_sq, -1]),
        };
        self.fc2.fwd(&self.fc1.fwd(&y).gelu("tanh"))
    }
}

pub struct VisionModel {
    cfg: VisionConfig,
    patch_embed: Lin,      // Conv(flat) == Linear 1536 -> 1152
    pos_embed: Tensor,     // [num_position_embeddings, hidden] f32
    blocks: Vec<Block>,
    merger: Merger,
    /// One merger per `deepstack_visual_indexes` entry, tapped after that block.
    deepstack: Vec<Merger>,
    device: Device,
}

impl VisionModel {
    pub fn load(dir: &std::path::Path, cfg: &Config, device: Device) -> Result<Self> {
        let vc = cfg
            .vision
            .clone()
            .ok_or_else(|| anyhow::anyhow!("config.json has no vision_config"))?;
        let stc = SafeTensors::open(dir, &[])?;
        let p = "model.visual";
        if !stc.has(&format!("{p}.patch_embed.proj.weight")) {
            bail!("checkpoint has no `model.visual.*` tensors — no vision tower");
        }

        // patch_embed.proj.weight: [hidden, 3, tp, ps, ps] -> [hidden, 1536] -> [1536, hidden]
        let pw = stc.get(&format!("{p}.patch_embed.proj.weight"), device, true, false)?;
        let indim = pw.numel() as i64 / vc.hidden_size;
        let patch_embed = Lin {
            w_t: pw.reshape([vc.hidden_size, indim]).transpose(0, 1).contiguous(),
            b: Some(
                stc.get(&format!("{p}.patch_embed.proj.bias"), device, true, false)?
                    .to_kind(Kind::Float),
            ),
        };

        let pos_embed = stc
            .get(&format!("{p}.pos_embed.weight"), device, false, false)?
            .to_kind(Kind::Float);

        let mut blocks = Vec::with_capacity(vc.depth as usize);
        for i in 0..vc.depth {
            let bk = format!("{p}.blocks.{i}");
            blocks.push(Block {
                norm1: Ln::load(&stc, &format!("{bk}.norm1"), device, vc.layernorm_eps)?,
                norm2: Ln::load(&stc, &format!("{bk}.norm2"), device, vc.layernorm_eps)?,
                qkv: Lin::load(&stc, &format!("{bk}.attn.qkv"), device, true)?,
                proj: Lin::load(&stc, &format!("{bk}.attn.proj"), device, true)?,
                fc1: Lin::load(&stc, &format!("{bk}.mlp.linear_fc1"), device, true)?,
                fc2: Lin::load(&stc, &format!("{bk}.mlp.linear_fc2"), device, true)?,
            });
        }

        Ok(Self {
            cfg: vc.clone(),
            patch_embed,
            pos_embed,
            blocks,
            merger: Merger::load(&stc, &format!("{p}.merger"), device, false)?,
            deepstack: (0..vc.deepstack_visual_indexes.len())
                .map(|i| {
                    Merger::load(&stc, &format!("{p}.deepstack_merger_list.{i}"), device, true)
                })
                .collect::<Result<_>>()?,
            device,
        })
    }

    /// `image_path` → spliceable `[n_tok, text_hidden]` embeddings + grid.
    pub fn embed_image(&self, image_path: &str) -> Result<ImageEmbeds> {
        let (patches, gt, gh, gw) = self.preprocess(image_path)?;
        let (emb, deepstack) = self.forward(&patches, gt, gh, gw);
        Ok(ImageEmbeds {
            embeddings: emb.to_kind(Kind::Half),
            deepstack: deepstack.into_iter().map(|t| t.to_kind(Kind::Half)).collect(),
            grid_t: gt,
            grid_h: gh,
            grid_w: gw,
        })
    }

    // --- preprocessing -----------------------------------------------------

    fn preprocess(&self, path: &str) -> Result<(Tensor, i64, i64, i64)> {
        let vc = &self.cfg;
        let img = image::open(path)?.to_rgb8();
        let (w0, h0) = (img.width() as i64, img.height() as i64);
        let factor = vc.patch_size * vc.spatial_merge_size;
        let (h1, w1) = smart_resize(h0, w0, factor, vc.min_pixels, vc.max_pixels)?;

        // bilinear resize on CPU via image crate, then to a [H,W,3] f32 tensor
        let resized = image::imageops::resize(
            &img,
            w1 as u32,
            h1 as u32,
            image::imageops::FilterType::CatmullRom, // Pillow BICUBIC ~ Catmull-Rom
        );
        let raw: Vec<f32> = resized.as_raw().iter().map(|&b| b as f32).collect();
        let t = Tensor::from_slice(&raw)
            .reshape([h1, w1, 3])
            .to_device(self.device)
            .to_kind(Kind::Float);
        // rescale 1/255, normalize
        let mean_v: Vec<f32> = vc.image_mean.iter().map(|&x| x as f32).collect();
        let std_v: Vec<f32> = vc.image_std.iter().map(|&x| x as f32).collect();
        let mean = Tensor::from_slice(&mean_v).to_device(self.device).reshape([1, 1, 3]);
        let std = Tensor::from_slice(&std_v).to_device(self.device).reshape([1, 1, 3]);
        let t = ((t / 255.0) - mean) / std; // [H,W,3]

        let ps = vc.patch_size;
        let tp = vc.temporal_patch_size;
        let sm = vc.spatial_merge_size;
        let (gh, gw) = (h1 / ps, w1 / ps);
        let grid_t = 1i64;

        // [H,W,3] -> [3,H,W] -> [tp,3,H,W] (tile a still image across the temporal patch)
        let chw = t.permute([2, 0, 1]).contiguous();
        let frames = chw.unsqueeze(0).expand([tp, 3, h1, w1], false).contiguous();
        // mirror the numpy reshape/transpose in Qwen3VLVisionModel.preprocess
        let p = frames.reshape([
            grid_t, tp, 3, gh / sm, sm, ps, gw / sm, sm, ps,
        ]);
        let p = p.permute([0, 3, 6, 4, 7, 2, 1, 5, 8]).contiguous();
        let flat = p.reshape([grid_t * gh * gw, 3 * tp * ps * ps]);
        Ok((flat.to_kind(Kind::Half), grid_t, gh, gw))
    }

    // --- tower forward ---------------------------------------------------

    /// Returns the merged image features and, when the tower has deepstack
    /// mergers, one extra feature block per tap.
    fn forward(&self, patches: &Tensor, gt: i64, gh: i64, gw: i64) -> (Tensor, Vec<Tensor>) {
        let _ng = tch::no_grad_guard();
        let vc = &self.cfg;
        let n = gt * gh * gw;

        let mut x = self.patch_embed.fwd(patches); // [n, hidden] f32
        x = &x + self.interp_pos_embed(gt, gh, gw); // + [n, hidden]

        // 2D rotary tables for vision attention: emb [n, head_dim/2]
        let (cos, sin) = self.rope_2d(gt, gh, gw); // each [n, head_dim] f32

        let nh = vc.num_heads;
        let hd = vc.head_dim;
        let scale = (hd as f64).powf(-0.5);

        let merge_sq = vc.spatial_merge_size * vc.spatial_merge_size;
        let mut deepstack = Vec::with_capacity(self.deepstack.len());
        for (bi, blk) in self.blocks.iter().enumerate() {
            // --- attention (non-causal, full) ---
            let h = blk.norm1.fwd(&x); // [n, hidden] f32
            let qkv = blk.qkv.fwd(&h).reshape([n, 3, nh, hd]); // [n,3,nh,hd]
            let q = qkv.select(1, 0);
            let k = qkv.select(1, 1);
            let v = qkv.select(1, 2);
            let q = apply_rope(&q, &cos, &sin); // [n, nh, hd]
            let k = apply_rope(&k, &cos, &sin);
            // [nh, n, hd]
            let q = q.transpose(0, 1).contiguous().to_kind(Kind::Half);
            let k = k.transpose(0, 1).contiguous().to_kind(Kind::Half);
            let v = v.transpose(0, 1).contiguous().to_kind(Kind::Half);
            let scores = q.matmul(&k.transpose(-2, -1)).to_kind(Kind::Float) * scale;
            let probs = scores.softmax(-1, Kind::Float).to_kind(Kind::Half);
            let attn = probs.matmul(&v); // [nh, n, hd] half
            let attn = attn.transpose(0, 1).contiguous().reshape([n, nh * hd]);
            let a_out = blk.proj.fwd(&attn); // [n, hidden] f32
            x = &x + a_out;

            // --- MLP ---
            let h = blk.norm2.fwd(&x);
            let m = blk.fc1.fwd(&h).gelu("tanh");
            let m = blk.fc2.fwd(&m);
            x = &x + m;

            // Deepstack taps read the block *output*, in the order the indexes
            // are listed — which is also the order the text layers consume them.
            if let Some(k) = vc
                .deepstack_visual_indexes
                .iter()
                .position(|&d| d == bi as i64)
            {
                deepstack.push(self.deepstack[k].fwd(&x, merge_sq));
            }
        }

        (self.merger.fwd(&x, merge_sq), deepstack)
    }

    /// `Qwen3VLPosEmbedding.fast_pos_embed_interpolate` for a single grid.
    fn interp_pos_embed(&self, gt: i64, gh: i64, gw: i64) -> Tensor {
        let dev = self.device;
        let side = (self.cfg.num_position_embeddings as f64).sqrt(); // 48
        let side_i = side as i64;
        let m = self.cfg.spatial_merge_size;

        let lin = |n: i64| {
            if n == 1 {
                Tensor::zeros([1], (Kind::Float, dev))
            } else {
                Tensor::arange(n, (Kind::Float, dev)) * ((side - 1.0) / (n - 1) as f64)
            }
        };
        let h_idx = lin(gh);
        let w_idx = lin(gw);
        let h_fl = h_idx.floor();
        let w_fl = w_idx.floor();
        let h_cl = (&h_fl + 1.0).clamp(0.0, side - 1.0);
        let w_cl = (&w_fl + 1.0).clamp(0.0, side - 1.0);
        let dh = (&h_idx - &h_fl).reshape([gh, 1]);
        let dw = (&w_idx - &w_fl).reshape([1, gw]);

        let base_h = (&h_fl * side).reshape([gh, 1]);
        let base_hc = (&h_cl * side).reshape([gh, 1]);
        let w_fl_r = w_fl.reshape([1, gw]);
        let w_cl_r = w_cl.reshape([1, gw]);

        let idx = [
            (&base_h + &w_fl_r).reshape([-1]),
            (&base_h + &w_cl_r).reshape([-1]),
            (&base_hc + &w_fl_r).reshape([-1]),
            (&base_hc + &w_cl_r).reshape([-1]),
        ];
        let idh = -&dh + 1.0;
        let idw = -&dw + 1.0;
        let wgt = [
            (&idh * &idw).reshape([-1, 1]),
            (&idh * &dw).reshape([-1, 1]),
            (&dh * &idw).reshape([-1, 1]),
            (&dh * &dw).reshape([-1, 1]),
        ];
        let mut pe = Tensor::zeros([gh * gw, self.cfg.hidden_size], (Kind::Float, dev));
        for i in 0..4 {
            let sel = self
                .pos_embed
                .index_select(0, &idx[i].to_kind(Kind::Int64).clamp(0, side_i * side_i - 1));
            pe = pe + sel * &wgt[i];
        }
        // repeat over t, then the merge-window permute -> [t*gh*gw, hidden]
        let pe = pe.repeat([gt, 1]);
        pe.reshape([gt, gh / m, m, gw / m, m, self.cfg.hidden_size])
            .permute([0, 1, 3, 2, 4, 5])
            .contiguous()
            .reshape([gt * gh * gw, self.cfg.hidden_size])
    }

    /// `qwen2_position_embedding_grid_2d` → NEOX cos/sin, each `[n, head_dim]`.
    fn rope_2d(&self, gt: i64, gh: i64, gw: i64) -> (Tensor, Tensor) {
        let dev = self.device;
        let m = self.cfg.spatial_merge_size;
        let hd = self.cfg.head_dim;

        // hpos/wpos ids in merge-window order, length gh*gw, repeated t times
        let hpos = Tensor::arange(gh, (Kind::Int64, dev))
            .reshape([gh, 1])
            .expand([gh, gw], false)
            .reshape([gh / m, m, gw / m, m])
            .permute([0, 2, 1, 3])
            .contiguous()
            .reshape([-1]);
        let wpos = Tensor::arange(gw, (Kind::Int64, dev))
            .reshape([1, gw])
            .expand([gh, gw], false)
            .reshape([gh / m, m, gw / m, m])
            .permute([0, 2, 1, 3])
            .contiguous()
            .reshape([-1]);
        let hpos = hpos.repeat([gt]);
        let wpos = wpos.repeat([gt]);

        // freqs: dim = head_dim/2 ; inv_freq over arange(0,dim,2)/dim
        let dim = hd / 2;
        let ar = Tensor::arange_start_step(0i64, dim, 2i64, (Kind::Float, dev)) / dim as f64;
        let inv_freq = (ar * (-self.cfg.rope_theta.ln())).exp(); // theta^(-x)
        // angles: [n, dim/2] for h and w, concatenated -> [n, dim] = [n, head_dim/2]
        let ah = hpos.to_kind(Kind::Float).reshape([-1, 1]).matmul(&inv_freq.reshape([1, -1]));
        let aw = wpos.to_kind(Kind::Float).reshape([-1, 1]).matmul(&inv_freq.reshape([1, -1]));
        let emb = Tensor::cat(&[ah, aw], -1); // [n, head_dim/2]
        let emb2 = Tensor::cat(&[&emb, &emb], -1); // [n, head_dim]
        (emb2.cos(), emb2.sin())
    }
}

/// NEOX rotate: `x*cos + rotate_half(x)*sin`. `x` `[n, nh, hd]`, cos/sin `[n, hd]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let hd = *x.size().last().unwrap();
    let x1 = x.narrow(-1, 0, hd / 2);
    let x2 = x.narrow(-1, hd / 2, hd / 2);
    let rot = Tensor::cat(&[&(-&x2), &x1], -1);
    let cos = cos.unsqueeze(1);
    let sin = sin.unsqueeze(1);
    x * cos + rot * sin
}

/// `qwen2_smart_resize` → `(h, w)`, both multiples of `factor`.
fn smart_resize(h: i64, w: i64, factor: i64, min_px: i64, max_px: i64) -> Result<(i64, i64)> {
    if h < factor || w < factor {
        bail!("image {w}x{h} smaller than patch factor {factor}");
    }
    let f = factor as f64;
    let round_f = |v: f64| ((v / f).round() * f) as i64;
    let mut hb = round_f(h as f64).max(factor);
    let mut wb = round_f(w as f64).max(factor);
    if hb * wb > max_px {
        let beta = ((h as f64 * w as f64) / max_px as f64).sqrt();
        hb = ((h as f64 / beta / f).floor() * f) as i64;
        wb = ((w as f64 / beta / f).floor() * f) as i64;
    } else if hb * wb < min_px {
        let beta = (min_px as f64 / (h as f64 * w as f64)).sqrt();
        hb = ((h as f64 * beta / f).ceil() * f) as i64;
        wb = ((w as f64 * beta / f).ceil() * f) as i64;
    }
    Ok((hb.max(factor), wb.max(factor)))
}

// ---------------------------------------------------------------------------
// MRoPE position ids + angle table for the text model
// ---------------------------------------------------------------------------

/// Port of `rope.cu::gen_mrope_pos_ids` for one image span. `seq_len` positions;
/// `img_span = (first_tok_idx, first_tok_idx + n_img_tokens)` marks the image
/// tokens in the sequence; `grid = (t, h, w)` is the *patch* grid (pre-merge).
/// Returns `pos_thw` `[3, seq_len]` i64 and the next free position.
pub fn mrope_pos_ids(
    seq_len: i64,
    img_span: (i64, i64),
    grid: (i64, i64, i64),
    merge: i64,
) -> (Vec<i64>, Vec<i64>, Vec<i64>, i64) {
    let (gt, gh, gw) = (grid.0, grid.1 / merge, grid.2 / merge);
    let (s0, s1) = img_span;
    let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
    let mut base_t = 0i64;
    let mut next_base = 0i64;
    for i in 0..seq_len {
        if i >= s0 && i < s1 {
            let k = i - s0;
            let kt = base_t + (k / gw / gh) % gt;
            let kh = base_t + (k / gw) % gh;
            let kw = base_t + k % gw;
            t.push(kt);
            h.push(kh);
            w.push(kw);
            next_base = next_base.max(kt + 1).max(kh + 1).max(kw + 1);
        } else {
            base_t = next_base;
            t.push(base_t);
            h.push(base_t);
            w.push(base_t);
            base_t += 1;
            next_base = base_t;
        }
    }
    (t, h, w, next_base)
}

/// Build the `[max_pos, rotary_dim/2]` f32 MRoPE angle table on `device`.
/// Rows `0..seq_len` use the 3-D `(t,h,w)` positions; rows past `seq_len`
/// continue as a plain contiguous ramp from `next_base` (decode is text-only).
/// `inv_freq` is the text model's 1-D rope frequency vector `[rotary_dim/2]`.
/// `section` is `[t, h, w]` — how many of the `rotary_dim/2` freq dims each axis
/// owns, assigned round-robin (`dim d → axis d % 3`).
pub fn mrope_angle_table(
    inv_freq: &Tensor,
    pos_thw: (&[i64], &[i64], &[i64]),
    next_base: i64,
    max_pos: i64,
    section: [i64; 3],
    device: Device,
) -> Tensor {
    let half = inv_freq.size()[inv_freq.dim() - 1];
    let seq_len = pos_thw.0.len() as i64;
    debug_assert_eq!(section.iter().sum::<i64>(), half);

    // full [3, max_pos] positions
    let mut t = pos_thw.0.to_vec();
    let mut h = pos_thw.1.to_vec();
    let mut w = pos_thw.2.to_vec();
    for i in seq_len..max_pos {
        let p = next_base + (i - seq_len);
        t.push(p);
        h.push(p);
        w.push(p);
    }
    let pos = Tensor::stack(
        &[
            Tensor::from_slice(&t),
            Tensor::from_slice(&h),
            Tensor::from_slice(&w),
        ],
        0,
    )
    .to_device(device)
    .to_kind(Kind::Float); // [3, max_pos]

    // axis index per freq dim: [0,1,2,0,1,2,...]
    let axis_v: Vec<i64> = (0..half).map(|j| j % 3).collect();
    let axis = Tensor::from_slice(&axis_v).to_device(device);
    let sel = pos.index_select(0, &axis); // [half, max_pos]
    let inv = inv_freq.to_device(device).to_kind(Kind::Float).reshape([half, 1]);
    (sel * inv).transpose(0, 1).contiguous() // [max_pos, half]
}

// ---------------------------------------------------------------------------
// End-to-end multimodal generate (Qwen3.5 + vision, greedy / temperature).
// `bin/infer` only needs to load the tower and call this.
// ---------------------------------------------------------------------------

pub struct VisionInferOpts<'a> {
    pub image_paths: &'a [String],
    pub prompt: &'a str,
    pub chat: bool,
    pub no_think: bool,
    pub max_new: usize,
    pub temperature: f64,
    pub top_k: i64,
    pub top_p: f64,
    pub max_seq_len: Option<i64>,
    pub timing: bool,
}

fn greedy_or_sample(logits: &Tensor, temp: f64, top_k: i64, top_p: f64) -> i64 {
    if temp <= 0.0 {
        return logits.argmax(0, false).int64_value(&[]);
    }
    let mut l = logits / temp;
    if top_k > 0 {
        let (vals, _) = l.topk(top_k.min(l.size()[0]), 0, true, true);
        let kth = vals.get(vals.size()[0] - 1).double_value(&[]);
        l = l.where_scalarother(&l.ge(kth), f64::NEG_INFINITY);
    }
    let mut probs = l.softmax(0, Kind::Float);
    if top_p < 1.0 {
        let (sorted, idx) = probs.sort(0, true);
        let cum = sorted.cumsum(0, Kind::Float);
        let mask = cum
            .le(top_p)
            .logical_or(&Tensor::arange(sorted.size()[0], (Kind::Int64, sorted.device())).eq(0));
        let kept = sorted.where_scalarother(&mask, 0.0);
        probs = probs.zeros_like().scatter(0, &idx, &kept);
        probs = &probs / probs.sum(Kind::Float);
    }
    probs.multinomial(1, true).int64_value(&[0])
}

/// Run one image + text prompt to completion, streaming to stdout.
pub fn run_infer(
    model: &Model,
    vision: &VisionModel,
    tok: &Tok,
    opts: &VisionInferOpts,
) -> Result<()> {
    let cfg = &model.config;
    let device = model.device();
    // The tower is the same for every Qwen3-VL-family checkpoint; what differs
    // downstream is the decoder, so dispatch on that rather than on the arch
    // string. `vision.is_some()` already implies the tower loaded.
    if cfg.vision.is_none() {
        bail!("--vision needs a checkpoint with a vision tower (no vision_config)");
    }
    let vc = cfg.vision.as_ref().unwrap();
    let merge = vc.spatial_merge_size;
    if opts.image_paths.len() != 1 {
        bail!("this build supports exactly one --vision image per prompt");
    }

    // --- prompt: split on the <image> marker ---
    let mut text = if opts.chat {
        Tok::qwen_chat_prompt(opts.prompt, None)
    } else {
        opts.prompt.to_string()
    };
    if opts.chat && opts.no_think {
        text.push_str("<think>\n\n</think>\n\n");
    }
    let (before, after) = match text.split_once("<image>") {
        Some((a, b)) => (a.to_string(), b.to_string()),
        None => (String::new(), text.clone()), // image goes first
    };
    let before_ids = tok.encode(&before)?;
    let after_ids = tok.encode(&after)?;

    // --- vision tower ---
    let t_v = Instant::now();
    eprintln!("running vision tower on {} ...", opts.image_paths[0]);
    let ie = vision.embed_image(&opts.image_paths[0])?;
    let n_img = ie.num_tokens();
    eprintln!(
        "  grid {}x{}x{} -> {} image tokens ({:.2}s)",
        ie.grid_t, ie.grid_h, ie.grid_w, n_img, t_v.elapsed().as_secs_f64()
    );

    // --- assemble the spliced embedding stream + id sequence ---
    let vs = cfg.vision_start_token_id;
    let ve = cfg.vision_end_token_id;
    let pad = cfg.image_token_id;
    let mut seq: Vec<i64> = Vec::new();
    seq.extend_from_slice(&before_ids);
    seq.push(vs);
    seq.extend(std::iter::repeat(pad).take(n_img as usize));
    seq.push(ve);
    seq.extend_from_slice(&after_ids);
    let seq_len = seq.len() as i64;
    let img_first = before_ids.len() as i64 + 1;
    let img_span = (img_first, img_first + n_img);

    // embedding stream: text embeddings for everything, image embeddings spliced
    // in for the placeholder span (build by concat so nothing is aliased).
    let emb_of = |slice: &[i64]| -> Tensor {
        if slice.is_empty() {
            Tensor::zeros([1, 0, cfg.hidden_size], (Kind::Half, device))
        } else {
            model
                .embed_tokens(&Tensor::from_slice(slice).reshape([1, -1]).to_device(device))
                .to_kind(Kind::Half)
        }
    };
    let img_emb = ie.embeddings.to_device(device).to_kind(Kind::Half).unsqueeze(0); // [1,n_img,h]
    let embeds = Tensor::cat(
        &[
            emb_of(&before_ids),
            emb_of(&[vs]),
            img_emb,
            emb_of(&[ve]),
            emb_of(&after_ids),
        ],
        1,
    ); // [1, seq_len, h]

    // --- MRoPE angle table ---
    let max_len = match opts.max_seq_len {
        Some(m) if m > 0 => m.max(seq_len + opts.max_new as i64 + 8),
        _ => seq_len + opts.max_new as i64 + 8,
    };
    let rope = crate::rope::RoPE::new(device, &cfg.rope);
    let half = rope.inv_freq.size()[0];
    let (pt, ph, pw, next_base) =
        mrope_pos_ids(seq_len, img_span, (ie.grid_t, ie.grid_h, ie.grid_w), merge);
    let section = cfg.mrope_section.unwrap_or([half, 0, 0]);
    let angle_table = mrope_angle_table(
        &rope.inv_freq,
        (&pt, &ph, &pw),
        next_base,
        max_len,
        section,
        device,
    );

    // Deepstack features are per-position, so they are widened here to a
    // full-sequence tensor that is zero outside the image span; the decoder then
    // adds them without needing to know where the image sits.
    let deepstack: Vec<Tensor> = ie
        .deepstack
        .iter()
        .map(|d| {
            let full = Tensor::zeros([1, seq_len, cfg.hidden_size], (Kind::Half, device));
            let _ = full
                .narrow(1, img_span.0, img_span.1 - img_span.0)
                .copy_(&d.unsqueeze(0).to_device(device));
            full
        })
        .collect();

    // --- prefill ---
    let eos = cfg.eos_token_ids.clone();
    // The hybrid stack carries GDN state and needs its own cache; every other
    // decoder in this family is the ordinary paged one.
    let hybrid = cfg.arch_kind.is_hybrid();
    let q35 = hybrid.then(|| Qwen35Cache::new(cfg, max_len, device));
    let paged = (!hybrid).then(|| crate::cache::PagedKvCache::new(cfg, max_len, device));
    let run = |x: &Tensor, ds: &[Tensor]| -> (Tensor, Tensor) {
        match (&q35, &paged) {
            (Some(c), _) => model.forward_qwen35_mm(x, c, Some(&angle_table), ds),
            (_, Some(c)) => model.forward_paged_mm(x, c, Some(&angle_table), ds),
            _ => unreachable!("exactly one cache is built"),
        }
    };
    let advance = |n: i64| match (&q35, &paged) {
        (Some(c), _) => c.advance(n),
        (_, Some(c)) => c.advance(n),
        _ => unreachable!(),
    };
    let t_p = Instant::now();
    let (_h, logits) = run(&embeds, &deepstack);
    advance(seq_len);
    let prefill_s = t_p.elapsed().as_secs_f64();
    let mut next = greedy_or_sample(
        &logits.select(0, 0).select(0, seq_len - 1),
        opts.temperature,
        opts.top_k,
        opts.top_p,
    );

    print!("{before}{after}");
    std::io::stdout().flush().ok();

    // --- decode ---
    let t_d = Instant::now();
    let mut generated = 0usize;
    while generated < opts.max_new {
        if eos.contains(&next) {
            break;
        }
        print!("{}", tok.decode(&[next])?);
        std::io::stdout().flush().ok();
        generated += 1;

        let x = model
            .embed_tokens(&Tensor::from_slice(&[next]).reshape([1, 1]).to_device(device))
            .to_kind(Kind::Half);
        // Decode steps carry no image tokens, so no deepstack.
        let (_h, l) = run(&x, &[]);
        advance(1);
        next = greedy_or_sample(&l.select(0, 0).select(0, 0), opts.temperature, opts.top_k, opts.top_p);
    }
    let decode_s = t_d.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "\x1b[2m{generated} tokens in {decode_s:.2}s — {:.1} tok/s\x1b[0m",
        generated as f64 / decode_s.max(1e-9)
    );
    if opts.timing {
        eprintln!(
            "vision+prefill: {seq_len} tok in {prefill_s:.3}s  |  decode: {generated} tok ({:.1} tok/s) [eager, mm]",
            generated as f64 / decode_s.max(1e-9)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident_merger(hidden: i64, merge_sq: i64, postshuffle: bool) -> Merger {
        let inw = hidden * merge_sq;
        // LayerNorm with unit weight and zero bias, and identity MLPs, so the
        // only thing the test measures is *where* the norm is applied.
        Merger {
            norm: Ln {
                w: Tensor::ones(
                    [if postshuffle { inw } else { hidden }],
                    (Kind::Float, Device::Cpu),
                ),
                b: Tensor::zeros(
                    [if postshuffle { inw } else { hidden }],
                    (Kind::Float, Device::Cpu),
                ),
                eps: 1e-6,
            },
            fc1: Lin { w_t: Tensor::eye(inw, (Kind::Half, Device::Cpu)), b: None },
            fc2: Lin { w_t: Tensor::eye(inw, (Kind::Half, Device::Cpu)), b: None },
            postshuffle,
        }
    }

    /// The deepstack mergers norm *after* the 2x2 shuffle, the final merger
    /// before it. Shapes are identical either way, and both produce plausible
    /// features, so getting it backwards is silent — the image features are
    /// merely normalized over the wrong group.
    #[test]
    fn postshuffle_norm_is_not_the_same_as_preshuffle() {
        let (hidden, merge_sq, n) = (8i64, 4i64, 8i64);
        // Give each patch in a group a different scale, so per-patch and
        // per-group normalization cannot coincide.
        let x = (Tensor::arange(n * hidden, (Kind::Float, Device::Cpu)) * 0.37)
            .sin()
            .reshape([n, hidden])
            * Tensor::from_slice(&[1.0f32, 4.0, 0.25, 2.0, 1.0, 4.0, 0.25, 2.0]).reshape([n, 1]);

        let pre = ident_merger(hidden, merge_sq, false).fwd(&x, merge_sq);
        let post = ident_merger(hidden, merge_sq, true).fwd(&x, merge_sq);
        assert_eq!(pre.size(), post.size());
        assert!(
            f64::try_from((&pre - &post).abs().max()).unwrap() > 1e-2,
            "pre- and post-shuffle norm gave the same answer; the test cannot tell them apart"
        );

        // Post-shuffle is exactly "group first, then normalize the whole row":
        // with identity projections the merger reduces to gelu(layernorm(group)).
        let grouped = x.reshape([n / merge_sq, -1]);
        let want = grouped
            .f_layer_norm(
                [hidden * merge_sq],
                None::<Tensor>,
                None::<Tensor>,
                1e-6,
                false,
            )
            .unwrap()
            .to_kind(Kind::Half)
            .gelu("tanh")
            .to_kind(Kind::Float);
        assert!(
            f64::try_from((&post - &want).abs().max()).unwrap() < 2e-2,
            "post-shuffle merger does not match group-then-normalize"
        );
    }

    /// Every patch must land in exactly one merged token, in order. An off-by-one
    /// in the shuffle silently scrambles the image's spatial layout.
    #[test]
    fn shuffle_groups_consecutive_patches() {
        let (hidden, merge_sq, n) = (4i64, 4i64, 8i64);
        let x = Tensor::arange(n * hidden, (Kind::Float, Device::Cpu)).reshape([n, hidden]);
        let grouped = x.reshape([n / merge_sq, -1]);
        assert_eq!(grouped.size(), vec![2, 16]);
        assert_eq!(
            Vec::<i64>::try_from(grouped.select(0, 0).to_kind(Kind::Int64)).unwrap(),
            (0..16).collect::<Vec<_>>()
        );
    }
}
