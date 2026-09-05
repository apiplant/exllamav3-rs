//! Port of `exllamav3/util/rope.py` (grade A for the scalar frequency math).
//! `inv_freq` / `attn_factor` are computed exactly as the Python `_rope_params_*`
//! methods; `apply()` forwards to the `rope` CUDA kernel like `RoPE.apply`.

use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy, PartialEq)]
pub enum RopeStyle {
    None = 0,
    Gptj = 1,
    Neox = 2,
}

#[derive(Clone)]
pub struct RopeSettings {
    pub head_dim: i64,
    pub rope_theta: f64,
    pub rotary_dim: Option<i64>,
    pub partial_rotary_factor: f64,
    pub max_position_embeddings: Option<i64>,
    pub original_max_position_embeddings: Option<i64>,
    pub rope_style: RopeStyle,
    /// raw `rope_scaling` dict from config.json, if any
    pub rope_scaling: Option<serde_json::Value>,
}

impl RopeSettings {
    pub fn rope_mode(&self) -> i32 {
        self.rope_style as i32
    }
}

pub struct RoPE {
    pub inv_freq: Tensor,
    pub attn_factor: f64,
    pub style: RopeStyle,
}

fn arange_step2_over_dim(dim: i64, device: Device) -> Tensor {
    // torch.arange(0, dim, 2).float() / dim
    Tensor::arange_start_step(0i64, dim, 2i64, (Kind::Float, device)) / (dim as f64)
}

impl RoPE {
    pub fn new(device: Device, rs: &RopeSettings) -> Self {
        let t = rs
            .rope_scaling
            .as_ref()
            .and_then(|s| s.get("rope_type").or_else(|| s.get("type")))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let (inv_freq, attn_factor) = match t.as_deref() {
            None | Some("default") | Some("mrope") => Self::default_params(device, rs),
            Some("linear") => {
                let (f, _) = Self::default_params(device, rs);
                let factor = Self::scaling_f64(rs, "factor", 1.0);
                (f / factor, 1.0)
            }
            Some("llama3") => Self::llama3_params(device, rs),
            Some("yarn") => Self::yarn_params(device, rs),
            Some("longrope") | Some("su") => Self::longrope_params(device, rs),
            Some("proportional") => Self::proportional_params(device, rs),
            Some(other) => panic!("Unknown rope_type: {other}"),
        };
        RoPE { inv_freq, attn_factor, style: rs.rope_style }
    }

    fn scaling_f64(rs: &RopeSettings, k: &str, default: f64) -> f64 {
        rs.rope_scaling
            .as_ref()
            .and_then(|s| s.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(default)
    }

    fn rotary_dim(rs: &RopeSettings) -> i64 {
        rs.rotary_dim
            .unwrap_or((rs.head_dim as f64 * rs.partial_rotary_factor) as i64)
    }

    fn default_params(device: Device, rs: &RopeSettings) -> (Tensor, f64) {
        let dim = Self::rotary_dim(rs);
        let inv_freq = 1.0 / rs.rope_theta.powf_tensor(&arange_step2_over_dim(dim, device));
        (inv_freq, 1.0)
    }

    fn llama3_params(device: Device, rs: &RopeSettings) -> (Tensor, f64) {
        let (inv_freq, _) = Self::default_params(device, rs);
        let factor = Self::scaling_f64(rs, "factor", 8.0);
        let low_freq_factor = Self::scaling_f64(rs, "low_freq_factor", 1.0);
        let high_freq_factor = Self::scaling_f64(rs, "high_freq_factor", 4.0);
        let old_ctx = rs
            .rope_scaling
            .as_ref()
            .and_then(|s| s.get("original_max_position_embeddings"))
            .and_then(|v| v.as_f64())
            .unwrap_or(8192.0);
        let low_wl = old_ctx / low_freq_factor;
        let high_wl = old_ctx / high_freq_factor;
        let wavelen = (2.0 * std::f64::consts::PI) / &inv_freq;
        let inv_freq_llama = wavelen.gt(low_wl).where_self(&(&inv_freq / factor), &inv_freq);
        let smooth = (old_ctx / &wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
        let smoothed = (&inv_freq_llama / factor) * (1.0 - &smooth)
            + inv_freq_llama.shallow_clone() * &smooth;
        let is_medium = wavelen.ge(high_wl).logical_and(&wavelen.le(low_wl));
        let inv_freq = is_medium.where_self(&smoothed, &inv_freq_llama);
        (inv_freq, 1.0)
    }

    fn proportional_params(device: Device, rs: &RopeSettings) -> (Tensor, f64) {
        let head_dim = rs.head_dim;
        let factor = Self::scaling_f64(rs, "factor", 1.0);
        let rope_angles = (rs.partial_rotary_factor * head_dim as f64 / 2.0) as i64;
        let inv_freq_rotated = 1.0
            / rs.rope_theta.powf_tensor(
                &(Tensor::arange_start_step(0i64, 2 * rope_angles, 2i64, (Kind::Float, device))
                    / head_dim as f64),
            );
        let nope = head_dim / 2 - rope_angles;
        let inv_freq = if nope > 0 {
            Tensor::cat(
                &[inv_freq_rotated, Tensor::zeros([nope], (Kind::Float, device))],
                0,
            )
        } else {
            inv_freq_rotated
        };
        (inv_freq / factor, 1.0)
    }

    fn yarn_inv_freq(device: Device, dim: i64, base: f64, rs: &RopeSettings, factor: f64, orig_max: i64) -> Tensor {
        let pos_freqs = base.powf_tensor(&arange_step2_over_dim(dim, device));
        let extrapolation = 1.0 / &pos_freqs;
        let sc = rs.rope_scaling.as_ref();
        let beta_fast = Self::scaling_f64(rs, "beta_fast", 32.0);
        let beta_slow = Self::scaling_f64(rs, "beta_slow", 1.0);
        let truncate = sc
            .and_then(|s| s.get("truncate"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let find = |num_rot: f64| {
            (dim as f64 * (orig_max as f64 / (num_rot * 2.0 * std::f64::consts::PI)).ln())
                / (2.0 * base.ln())
        };
        let (mut low, mut high) = (find(beta_fast), find(beta_slow));
        if truncate {
            low = low.floor();
            high = high.ceil();
        }
        low = low.max(0.0);
        high = high.min(dim as f64 - 1.0);
        if low == high {
            high += 0.001;
        }
        let linear = (Tensor::arange(dim / 2, (Kind::Float, device)) - low) / (high - low);
        let extrap_factor = 1.0 - linear.clamp(0.0, 1.0);
        let interpolation = 1.0 / (factor * &pos_freqs);
        interpolation * (1.0 - &extrap_factor) + extrapolation * &extrap_factor
    }

    fn yarn_params(device: Device, rs: &RopeSettings) -> (Tensor, f64) {
        let max_pos = rs
            .max_position_embeddings
            .expect("YaRN needs max_position_embeddings");
        let base = rs.rope_theta;
        let dim = Self::rotary_dim(rs);
        let sc = rs.rope_scaling.as_ref().unwrap();
        let orig = sc
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_i64());
        let (orig_max, factor) = match orig {
            Some(o) => (o, max_pos as f64 / o as f64),
            None => (max_pos, Self::scaling_f64(rs, "factor", 1.0)),
        };
        let attn_factor = sc.get("attention_factor").and_then(|v| v.as_f64()).unwrap_or_else(|| {
            let get_mscale = |scale: f64, mscale: f64| {
                if scale <= 1.0 {
                    1.0
                } else {
                    0.1 * mscale * scale.ln() + 1.0
                }
            };
            let mscale = sc.get("mscale").and_then(|v| v.as_f64());
            let mscale_all = sc.get("mscale_all_dim").and_then(|v| v.as_f64());
            match (mscale, mscale_all) {
                _ => get_mscale(factor, 1.0),
            }
        });
        (
            Self::yarn_inv_freq(device, dim, base, rs, factor, orig_max),
            attn_factor,
        )
    }

    fn longrope_params(device: Device, rs: &RopeSettings) -> (Tensor, f64) {
        let base = rs.rope_theta;
        let dim = Self::rotary_dim(rs);
        let a = rs.max_position_embeddings.unwrap();
        let sc = rs.rope_scaling.as_ref().unwrap();
        let b = sc
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_i64())
            .or(rs.original_max_position_embeddings)
            .unwrap();
        let key = if a > b { "long_factor" } else { "short_factor" };
        let factors: Vec<f64> = sc[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let ext = Tensor::from_slice(&factors).to_device(device).to_kind(Kind::Float);
        let scaling = if a > b {
            (1.0 + (a as f64 / b as f64).ln() / (b as f64).ln()).sqrt()
        } else {
            1.0
        };
        let inv_freq = 1.0 / (&ext * base.powf_tensor(&arange_step2_over_dim(dim, device)));
        (inv_freq, scaling)
    }

    /// `RoPE.apply` — fused optional Q/K RMSNorm + rotary embedding, in place.
    ///
    /// `freq_table`, when given, is a `[max_pos, rotary_dim/2]` f32 device tensor
    /// of **precomputed rotation angles** (`angle[pos][i] = pos * inv_freq[i]` for
    /// plain RoPE, or the 3D-interleaved MRoPE angle for a multimodal sequence).
    /// The kernel then indexes it by `positions[b] + token_pos` instead of
    /// deriving the angle from the scalar `inv_freq` vector — this is how both the
    /// vision tower's 2D RoPE and the text model's MRoPE are expressed. `None`
    /// keeps the ordinary 1-D `inv_freq` path (bit-identical to before).
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        position: i64,
        positions: Option<&Tensor>,
        q_norm: Option<&Tensor>,
        k_norm: Option<&Tensor>,
        norm_eps: f32,
        norm_constant_bias: f32,
        freq_table: Option<&Tensor>,
    ) {
        crate::ffi::rope(
            q,
            q,
            Some(k),
            Some(k),
            freq_table.unwrap_or(&self.inv_freq),
            position,
            positions,
            q_norm,
            k_norm,
            norm_eps,
            norm_constant_bias,
            self.style as i32,
            self.attn_factor as f32,
        );
    }

    /// Norm + rotate a single tensor in place, with no companion K.
    ///
    /// [`RoPE::apply`] always hands the kernel a K tensor, so it cannot be used
    /// to rotate one stream on its own: passing the same tensor for both leaves
    /// K with a null norm weight and rotates it twice. DFlash2 fills its draft
    /// cache from projected target states and needs exactly K, hence this.
    pub fn apply_one(
        &self,
        x: &Tensor,
        position: i64,
        positions: Option<&Tensor>,
        norm: Option<&Tensor>,
        norm_eps: f32,
        norm_constant_bias: f32,
    ) {
        crate::ffi::rope(
            x,
            x,
            None,
            None,
            &self.inv_freq,
            position,
            positions,
            norm,
            None,
            norm_eps,
            norm_constant_bias,
            self.style as i32,
            self.attn_factor as f32,
        );
    }

    /// Build the plain-RoPE angle table `[max_pos, rotary_dim/2]` f32 on `device`
    /// (`angle[p][i] = p * inv_freq[i]`). Multimodal position logic patches rows
    /// `0..prompt_len` afterwards; the tail is the ordinary contiguous ramp.
    pub fn angle_table(&self, max_pos: i64, device: Device) -> Tensor {
        let pos = Tensor::arange(max_pos, (Kind::Float, device)).reshape([max_pos, 1]);
        let inv = self.inv_freq.to_device(device).reshape([1, -1]);
        pos.matmul(&inv) // [max_pos, rot/2]
    }

    /// `rotary_dim / 2` — the width of one angle-table row.
    pub fn half_rotary_dim(&self) -> i64 {
        self.inv_freq.size()[self.inv_freq.dim() - 1]
    }
}

/// `base ** exponent` where exponent is a tensor (torch `base ** t`).
trait PowfTensor {
    fn powf_tensor(self, t: &Tensor) -> Tensor;
}
impl PowfTensor for f64 {
    fn powf_tensor(self, t: &Tensor) -> Tensor {
        (t * self.ln()).exp()
    }
}
