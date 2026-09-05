"""Build a tiny synthetic Qwen3-VL checkpoint (text stack + vision tower).

Random weights, real tensor names and shapes. The point is the *pairing*: the
Qwen3 block and the Qwen3-VL tower are each exercised elsewhere, but the
deepstack path between them -- tower taps, patch mergers with post-shuffle norm,
and the per-layer injection into the text residual -- only exists when both are
loaded together.

Usage: python3 tests/make_qwen3vl_fixture.py <out_dir>
"""
import json, os, sys
import torch
from safetensors.torch import save_file

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen3vl_fixture"
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)

# text side — head_dim 128, the paged attention kernel is built for wide heads
H, NQ, NKV, HD = 512, 4, 2, 128
LAYERS, INTERM, VOCAB = 4, 256, 512
# vision side
VH, VDEPTH, VHEADS, PATCH, TPATCH, MERGE = 128, 4, 4, 16, 2, 2
NPOS = 64                     # 8x8 position grid
DEEPSTACK = [1, 2]            # tap after vision blocks 1 and 2
VINTERM = 256

IMG_TOK, VS_TOK, VE_TOK = 200, 201, 202

cfg = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": IMG_TOK,
    "vision_start_token_id": VS_TOK,
    "vision_end_token_id": VE_TOK,
    "text_config": {
        "vocab_size": VOCAB,
        "hidden_size": H,
        "intermediate_size": INTERM,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": NQ,
        "num_key_value_heads": NKV,
        "head_dim": HD,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "eos_token_id": 3,
        "bos_token_id": 1,
        "rope_theta": 10000.0,
        "rope_scaling": {"rope_type": "default", "mrope_section": [32, 16, 16],
                         "mrope_interleaved": True},
    },
    "vision_config": {
        "depth": VDEPTH,
        "hidden_size": VH,
        "intermediate_size": VINTERM,
        "num_heads": VHEADS,
        "out_hidden_size": H,
        "patch_size": PATCH,
        "temporal_patch_size": TPATCH,
        "spatial_merge_size": MERGE,
        "num_position_embeddings": NPOS,
        "deepstack_visual_indexes": DEEPSTACK,
        "model_type": "qwen3_vl",
    },
}
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
json.dump({
    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    "min_pixels": 256, "max_pixels": 65536,
}, open(os.path.join(out, "preprocessor_config.json"), "w"), indent=2)

t = {}
def w(name, *shape):
    t[name] = (torch.randn(*shape) * 0.05).to(torch.float16)
def z(name, *shape):
    t[name] = torch.zeros(*shape, dtype=torch.float16)
def one(name, *shape):
    # Norm *weights* must be ~1, not 0. This port adds a constant bias of 1.0
    # only where the architecture stores its RMSNorm weights zero-centred
    # (Qwen3.5, qwen4_exp); a Qwen3 RMSNorm and every LayerNorm use the stored
    # weight as-is, so a zero weight silently zeroes the whole model -- which
    # still passes a finite-logits check.
    t[name] = (torch.ones(*shape) + torch.randn(*shape) * 0.02).to(torch.float16)

# --- text stack ---
P = "model.language_model"
w(f"{P}.embed_tokens.weight", VOCAB, H)
one(f"{P}.norm.weight", H)
w("lm_head.weight", VOCAB, H)
for i in range(LAYERS):
    k = f"{P}.layers.{i}"
    one(f"{k}.input_layernorm.weight", H)
    one(f"{k}.post_attention_layernorm.weight", H)
    w(f"{k}.self_attn.q_proj.weight", NQ * HD, H)
    w(f"{k}.self_attn.k_proj.weight", NKV * HD, H)
    w(f"{k}.self_attn.v_proj.weight", NKV * HD, H)
    w(f"{k}.self_attn.o_proj.weight", H, NQ * HD)
    one(f"{k}.self_attn.q_norm.weight", HD)
    one(f"{k}.self_attn.k_norm.weight", HD)
    w(f"{k}.mlp.gate_proj.weight", INTERM, H)
    w(f"{k}.mlp.up_proj.weight", INTERM, H)
    w(f"{k}.mlp.down_proj.weight", H, INTERM)

# --- vision tower ---
V = "model.visual"
w(f"{V}.patch_embed.proj.weight", VH, 3, TPATCH, PATCH, PATCH)
z(f"{V}.patch_embed.proj.bias", VH)
w(f"{V}.pos_embed.weight", NPOS, VH)
for i in range(VDEPTH):
    b = f"{V}.blocks.{i}"
    for n in ("norm1", "norm2"):
        one(f"{b}.{n}.weight", VH); z(f"{b}.{n}.bias", VH)
    w(f"{b}.attn.qkv.weight", 3 * VH, VH); z(f"{b}.attn.qkv.bias", 3 * VH)
    w(f"{b}.attn.proj.weight", VH, VH);    z(f"{b}.attn.proj.bias", VH)
    w(f"{b}.mlp.linear_fc1.weight", VINTERM, VH); z(f"{b}.mlp.linear_fc1.bias", VINTERM)
    w(f"{b}.mlp.linear_fc2.weight", VH, VINTERM); z(f"{b}.mlp.linear_fc2.bias", VH)

def merger(key, in_dim):
    # The final merger norms per patch (VH wide); the deepstack mergers norm the
    # concatenated 2x2 group (VH * merge**2 wide). Same MLP either way.
    one(f"{key}.norm.weight", in_dim); z(f"{key}.norm.bias", in_dim)
    w(f"{key}.linear_fc1.weight", VH * MERGE * MERGE, VH * MERGE * MERGE)
    z(f"{key}.linear_fc1.bias", VH * MERGE * MERGE)
    w(f"{key}.linear_fc2.weight", H, VH * MERGE * MERGE)
    z(f"{key}.linear_fc2.bias", H)

merger(f"{V}.merger", VH)
for i in range(len(DEEPSTACK)):
    merger(f"{V}.deepstack_merger_list.{i}", VH * MERGE * MERGE)

save_file(t, os.path.join(out, "model.safetensors"))
print(f"{out}: {len(t)} tensors, {sum(v.numel() * v.element_size() for v in t.values()) / 1e6:.1f} MB")
