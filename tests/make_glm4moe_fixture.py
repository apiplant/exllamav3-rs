"""Build a tiny synthetic Glm4Moe (GLM-4.5/4.6, SolarOpen) checkpoint.

Random weights, real *shapes* and real tensor names. The smallest published
Glm4Moe exl3 quant is ~27 GB (GLM-4.5-Air at 2.0bpw, or the REAP-82B at
2.5bpw), which does not fit a 24 GB card, so this arch cannot be validated
against a real checkpoint here. This fixture exists to run the code paths that
are unique to it -- the `dots` router with its selection bias, the leading dense
layers, the ungated shared expert -- rather than to check numerics.

Usage: python3 tests/make_glm4moe_fixture.py <out_dir>
"""
import json, os, sys
import torch
from safetensors.torch import save_file

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/glm4moe_fixture"
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)

# head_dim 128: the paged attention kernel is built for wide heads, so a toy
# fixture still has to use a real head size.
H = 512
NQ, NKV, HD = 4, 2, 128      # ratio 2, no KV repeat needed
LAYERS = 4
FIRST_DENSE = 1              # layer 0 dense, 1..3 sparse
VOCAB = 256
INTERM = 256                 # dense MLP width
MOE_I, NEXP, TOPK, NSHARED = 128, 8, 3, 1
SCALE = 2.5

cfg = {
    "architectures": ["Glm4MoeForCausalLM"],
    "vocab_size": VOCAB,
    "hidden_size": H,
    "intermediate_size": INTERM,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": NQ,
    "num_key_value_heads": NKV,
    "head_dim": HD,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "eos_token_id": 2,
    "bos_token_id": 1,
    "rope_theta": 10000.0,
    "partial_rotary_factor": 0.5,
    "use_qk_norm": True,
    # moe
    "n_routed_experts": NEXP,
    "num_experts_per_tok": TOPK,
    "moe_intermediate_size": MOE_I,
    "n_shared_experts": NSHARED,
    "first_k_dense_replace": FIRST_DENSE,
    "routed_scaling_factor": SCALE,
    # MTP layers exist in the real checkpoints and are simply never loaded
    "num_nextn_predict_layers": 1,
}
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

t = {}
def w(name, *shape):
    t[name] = (torch.randn(*shape) * 0.05).to(torch.float16)
def z(name, *shape):
    t[name] = torch.zeros(*shape, dtype=torch.float16)
def one(name, *shape):
    # Glm4Moe stores RMSNorm weights around 1 and this port applies them as-is
    # (no constant bias), so zero weights would zero the model's output while
    # still passing every finite/shape check.
    t[name] = (torch.ones(*shape) + torch.randn(*shape) * 0.02).to(torch.float16)

w("model.embed_tokens.weight", VOCAB, H)
one("model.norm.weight", H)
w("lm_head.weight", VOCAB, H)

for i in range(LAYERS):
    k = f"model.layers.{i}"
    one(f"{k}.input_layernorm.weight", H)
    one(f"{k}.post_attention_layernorm.weight", H)
    w(f"{k}.self_attn.q_proj.weight", NQ * HD, H)
    w(f"{k}.self_attn.k_proj.weight", NKV * HD, H)
    w(f"{k}.self_attn.v_proj.weight", NKV * HD, H)
    w(f"{k}.self_attn.o_proj.weight", H, NQ * HD)
    one(f"{k}.self_attn.q_norm.weight", HD)
    one(f"{k}.self_attn.k_norm.weight", HD)
    if i < FIRST_DENSE:
        w(f"{k}.mlp.gate_proj.weight", INTERM, H)
        w(f"{k}.mlp.up_proj.weight", INTERM, H)
        w(f"{k}.mlp.down_proj.weight", H, INTERM)
    else:
        w(f"{k}.mlp.gate.weight", NEXP, H)
        # stored fp32 upstream, which is what triggers the recentering path
        t[f"{k}.mlp.gate.e_score_correction_bias"] = torch.randn(NEXP)
        for e in range(NEXP):
            w(f"{k}.mlp.experts.{e}.gate_proj.weight", MOE_I, H)
            w(f"{k}.mlp.experts.{e}.up_proj.weight", MOE_I, H)
            w(f"{k}.mlp.experts.{e}.down_proj.weight", H, MOE_I)
        w(f"{k}.mlp.shared_experts.gate_proj.weight", MOE_I * NSHARED, H)
        w(f"{k}.mlp.shared_experts.up_proj.weight", MOE_I * NSHARED, H)
        w(f"{k}.mlp.shared_experts.down_proj.weight", H, MOE_I * NSHARED)

# An MTP layer the loader must ignore: `num_hidden_layers` stays at the trunk
# depth, so these tensors are present in the file and never read.
mk = f"model.layers.{LAYERS}"
one(f"{mk}.input_layernorm.weight", H)
w(f"{mk}.eh_proj.weight", H, 2 * H)

save_file(t, os.path.join(out, "model.safetensors"))
print(f"{out}: {len(t)} tensors, {sum(v.numel() * v.element_size() for v in t.values()) / 1e6:.1f} MB")
