"""Build a tiny synthetic Qwen4Exp (qwen3.8-flash-next) checkpoint.

Random weights, real *shapes* and real tensor names. There is no public
qwen4_exp exl3 checkpoint to test against, so this exists to prove the wiring
holds together end to end -- config parsing, every tensor key the loaders ask
for, the hybrid layer schedule, the PLE/n-gram path and the QSA mask -- not to
check numerics against a reference.

Usage: python3 tests/make_qwen4_fixture.py <out_dir>
"""
import json, os, sys
import torch
from safetensors.torch import save_file

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen4_fixture"
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)

H = 128           # hidden
NQ, NKV, HD = 8, 4, 32
LAYERS = 4        # full, linear, linear, full
VOCAB = 256
MOE_I, NEXP, TOPK = 64, 4, 2
HC = 4
IDX_H, IDX_HD, BUDGET, CR = 2, 32, 16, 4
NGRAM, HPN, PLE_CONV = 3, 2, 4
NUM_HEADS = (NGRAM - 1) * HPN
ROW_DIM = 160
PLE_EMB = NUM_HEADS * ROW_DIM
VOCAB_PER_HEAD = 97
EOS = 3

# GDN
LK, LV, LKD, LVD, CONV_K = 2, 4, 32, 32, 4
KD, VD = LK * LKD, LV * LVD
FDIM = 2 * KD + VD

layer_types = ["full_attention", "linear_attention", "linear_attention", "full_attention"]

cfg = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "text_config": {
        "vocab_size": VOCAB,
        "hidden_size": H,
        "intermediate_size": MOE_I,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": NQ,
        "num_key_value_heads": NKV,
        "head_dim": HD,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "eos_token_id": EOS,
        "bos_token_id": 1,
        "layer_types": layer_types,
        "attn_output_gate": True,
        "output_gate_type": "sigmoid",
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        # gdn
        "linear_conv_kernel_dim": CONV_K,
        "linear_num_key_heads": LK,
        "linear_num_value_heads": LV,
        "linear_key_head_dim": LKD,
        "linear_value_head_dim": LVD,
        # moe
        "num_experts": NEXP,
        "num_experts_per_tok": TOPK,
        "moe_intermediate_size": MOE_I,
        "shared_expert_intermediate_size": MOE_I,
        # hyper-connections
        "hc_count": HC,
        # qsa
        "indexer_n_heads": IDX_H,
        "indexer_head_dim": IDX_HD,
        "indexer_budget": BUDGET,
        "indexer_compress_ratio": CR,
        # ple / n-gram
        "ple_layer_ids": [2],          # 1-based -> layer index 1
        "ple_embed_dim": PLE_EMB,
        "ple_conv_kernel_size": PLE_CONV,
        "ngram_size": NGRAM,
        "heads_per_ngram": HPN,
    },
}
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

t = {}
def w(name, *shape):
    t[name] = (torch.randn(*shape) * 0.05).to(torch.float16)
def z(name, *shape):
    t[name] = torch.zeros(*shape, dtype=torch.float16)

P = "model.language_model"
w(f"{P}.embed_tokens.weight", VOCAB, H)
w("lm_head.weight", VOCAB, H)

def hc_site(key, combine=True):
    rank = 16
    z(f"{key}.hc_norm.weight", HC * H)
    w(f"{key}.input_mix_weight_down.weight", rank, HC * H)
    w(f"{key}.input_mix_weight_up.weight", HC * H, rank)
    if combine:
        w(f"{key}.block_inject_weight.weight", HC, HC * H)

for i, lt in enumerate(layer_types):
    k = f"{P}.layers.{i}"
    hc_site(f"{k}.attn_hyper_connection")
    hc_site(f"{k}.mlp_hyper_connection")
    if lt == "full_attention":
        w(f"{k}.self_attn.q_proj.weight", NQ * HD * 2, H)   # interleaved [q|gate]
        w(f"{k}.self_attn.k_proj.weight", NKV * HD, H)
        w(f"{k}.self_attn.v_proj.weight", NKV * HD, H)
        w(f"{k}.self_attn.o_proj.weight", H, NQ * HD)
        z(f"{k}.self_attn.q_norm.weight", HD)
        z(f"{k}.self_attn.k_norm.weight", HD)
        w(f"{k}.self_attn.indexer.index_qk_proj.weight", (IDX_H + 1) * IDX_HD, H)
        z(f"{k}.self_attn.indexer.q_layernorm.weight", IDX_HD)
        z(f"{k}.self_attn.indexer.k_layernorm.weight", IDX_HD)
    else:
        w(f"{k}.linear_attn.in_proj_qkv.weight", FDIM, H)
        w(f"{k}.linear_attn.in_proj_z.weight", VD, H)
        w(f"{k}.linear_attn.in_proj_b.weight", LV, H)
        w(f"{k}.linear_attn.in_proj_a.weight", LV, H)
        w(f"{k}.linear_attn.out_proj.weight", H, VD)
        w(f"{k}.linear_attn.conv1d.weight", FDIM, 1, CONV_K)
        t[f"{k}.linear_attn.A_log"] = torch.zeros(LV, dtype=torch.float16)
        t[f"{k}.linear_attn.dt_bias"] = torch.zeros(LV, dtype=torch.float16)
        z(f"{k}.linear_attn.norm.weight", LVD)
    # moe
    w(f"{k}.mlp.gate.weight", NEXP, H)
    for e in range(NEXP):
        w(f"{k}.mlp.experts.{e}.gate_proj.weight", MOE_I, H)
        w(f"{k}.mlp.experts.{e}.up_proj.weight", MOE_I, H)
        w(f"{k}.mlp.experts.{e}.down_proj.weight", H, MOE_I)
    w(f"{k}.mlp.shared_expert.gate_proj.weight", MOE_I, H)
    w(f"{k}.mlp.shared_expert.up_proj.weight", MOE_I, H)
    w(f"{k}.mlp.shared_expert.down_proj.weight", H, MOE_I)
    w(f"{k}.mlp.shared_expert_gate.weight", 1, H)
    if i == 1:
        w(f"{k}.ple.key_proj.weight", HC * H, PLE_EMB)
        w(f"{k}.ple.value_proj.weight", H, PLE_EMB)
        z(f"{k}.ple.norm_key.weight", HC * H)
        z(f"{k}.ple.norm_query.weight", HC * H)
        z(f"{k}.ple.norm_conv.weight", HC * H)
        w(f"{k}.ple.conv1d.weight", HC * H, 1, PLE_CONV)

hc_site(f"{P}.hyper_connection_mixer", combine=False)

# n-gram table: one row per (head, hash slot), plus the hash parameters
w(f"{P}.ple_embedding.weight", NUM_HEADS * VOCAB_PER_HEAD, ROW_DIM)
t[f"{P}.ngram_heads_offsets"] = (torch.arange(NUM_HEADS) * VOCAB_PER_HEAD).to(torch.int64)
t[f"{P}.ngram_heads_vocab_sizes"] = torch.full((NUM_HEADS,), VOCAB_PER_HEAD, dtype=torch.int64)
t[f"{P}.layer_multipliers"] = torch.tensor(
    [0x9E3779B1 + i * 0x01000193 for i in range(NGRAM)], dtype=torch.int64)

save_file(t, os.path.join(out, "model.safetensors"))
print(f"{out}: {len(t)} tensors, {sum(v.numel() * v.element_size() for v in t.values()) / 1e6:.1f} MB")
