"""Reference inference driver against upstream exllamav3, with the same CLI as
`cargo run --bin infer` so results can be compared directly.

    python py-infer.py --model ../models/Qwen3-0.6B-exl3-8.0bpw_H8 \
        --prompt "Name three primary colors." --chat --max-new 64

    # Qwen3.5 with the MTP draft head + an image:
    python py-infer.py --model /path/qwen3.5-27b --chat --mtp \
        --vision cat.png --prompt "<image>\nDescribe this." --max-new 200
"""

import sys, os, time, argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3"))


def log(m):
    print(f"[{time.time():.1f}] {m}", file=sys.stderr, flush=True)


p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="model directory")
p.add_argument("--prompt", required=True)
p.add_argument("--chat", action="store_true", help="wrap prompt in the Qwen chat template")
p.add_argument("--max-new", type=int, default=64)
p.add_argument("--temperature", type=float, default=0.0)
p.add_argument("--top-k", type=int, default=0)
p.add_argument("--top-p", type=float, default=1.0)
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--timing", action="store_true", help="print prefill / decode timing to stderr")
# --- parity with the Rust infer CLI / tabbyAPI config.yml knobs ---
p.add_argument("--max-seq-len", type=int, default=None, help="context length (cache size)")
p.add_argument("--rope-scale", type=float, default=None, help="linear RoPE position scaling")
p.add_argument("--rope-alpha", type=float, default=None, help="NTK-aware RoPE base scaling")
p.add_argument("--chunk-size", type=int, default=0, help="prefill chunk size (0 = default)")
p.add_argument("--cache-mode", default="FP16", help="FP16 | Q8 | Q6 | Q4")
p.add_argument("--cache-bits", type=int, default=0, help="alias for --cache-mode (0 = fp16)")
p.add_argument("--reasoning", type=lambda s: s.lower() not in ("0", "false", "no"), default=True)
p.add_argument("--no-think", action="store_true", help="append an empty <think></think> block")
p.add_argument("--mtp", action="store_true", help="load the MTP head as a draft model (Qwen3.5)")
p.add_argument("--vision", action="append", default=[], help="image file; <image> in the prompt marks the spot")
args = p.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

log("importing torch")
import torch

log(f"torch {torch.__version__} cuda={torch.version.cuda} dev={torch.cuda.get_device_name(0)}")

log("importing exllamav3")
from exllamav3 import (
    Config, Model, Cache, Tokenizer, Generator, Job,
    GreedySampler, ComboSampler, CacheLayer_quant,
)

log(f"exllamav3 from {__import__('exllamav3').__file__}")

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.model)) \
    if not os.path.isabs(args.model) else args.model


def cache_bits():
    if args.cache_bits:
        return args.cache_bits
    return {"FP16": 0, "F16": 0, "Q8": 8, "Q6": 6, "Q4": 4, "Q3": 3, "Q2": 2}[args.cache_mode.upper()]


log("Config.from_directory")
config = Config.from_directory(model_dir)

# RoPE overrides (tabby rope_alpha / rope_scale) — upstream RoPE has no built-in
# knob, so patch rope_settings before the model loads.
rs = config.rope_settings
if args.rope_alpha and args.rope_alpha != 1.0:
    d = int(round(rs.head_dim * rs.partial_rotary_factor)) or rs.head_dim
    rs.rope_theta *= args.rope_alpha ** (d / (d - 2))
if args.rope_scale and args.rope_scale != 1.0:
    rs.rope_scaling = {"rope_type": "linear", "factor": args.rope_scale}
    rs.override_type = "linear"

cache_size = args.max_seq_len or 4096

log("Model.from_config")
model = Model.from_config(config)

# With an MTP draft the target's recurrent (GDN) layers need per-token history
# planes so a rejected draft can be rewound (num_draft_tokens + 1).
mh = dict(max_history=5, max_batch_size=1) if args.mtp else {}
bits = cache_bits()
if bits:
    log(f"cache: {bits}-bit quantized")
    cache = Cache(model, max_num_tokens=cache_size, layer_type=CacheLayer_quant,
                  k_bits=bits, v_bits=bits, **mh)
else:
    cache = Cache(model, max_num_tokens=cache_size, **mh)

log("model.load")
model.load(progressbar=True)
log("model loaded")
tokenizer = Tokenizer.from_config(config)

# --- vision component ---
image_embeddings = []
if args.vision:
    from PIL import Image
    log("loading vision component")
    vision_model = Model.from_config(config, component="vision")
    vision_model.load(progressbar=True)
    for path in args.vision:
        ie = vision_model.get_image_embeddings(tokenizer=tokenizer, image=Image.open(path))
        image_embeddings.append(ie)

# --- MTP draft model ---
draft_model = None
draft_cache = None
if args.mtp:
    log("loading MTP head")
    draft_model = Model.from_config(config, component="mtp")
    draft_cache = Cache(draft_model, max_num_tokens=cache_size)  # MTP has 1 attn layer
    draft_model.load(progressbar=True)

# --- prompt ---
raw = args.prompt
for ie in image_embeddings:
    raw = raw.replace("<image>", ie.text_alias, 1)

if args.chat:
    prompt = f"<|im_start|>user\n{raw}<|im_end|>\n<|im_start|>assistant\n"
    if args.no_think or not args.reasoning:
        prompt += "<think>\n\n</think>\n\n"
else:
    prompt = raw

if args.temperature <= 0.0:
    sampler = GreedySampler()
else:
    sampler = ComboSampler(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

log("Generator")
gen_kwargs = dict(model=model, cache=cache, tokenizer=tokenizer)
if draft_model is not None:
    gen_kwargs.update(draft_model=draft_model, draft_cache=draft_cache)
generator = Generator(**gen_kwargs)

n_prompt = len(tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True,
                                embeddings=image_embeddings or None)[0])

log("generate")
job = Job(
    input_ids=tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True,
                               embeddings=image_embeddings or None),
    max_new_tokens=args.max_new,
    sampler=sampler,
    stop_conditions=[tokenizer.eos_token_id, "<|im_end|>"],
    embeddings=image_embeddings or None,
)
generator.enqueue(job)

print(prompt, end="", flush=True)
t0 = time.time()
final = {}
while generator.num_remaining_jobs():
    for r in generator.iterate():
        if r["stage"] == "streaming" and "text" in r:
            print(r["text"], end="", flush=True)
        if r["stage"] == "streaming" and r.get("eos"):
            final = r
dt = time.time() - t0
print()
log("done")

n_gen = final.get("new_tokens", 0)
n_prompt = final.get("prompt_tokens", n_prompt)
t_prefill = final.get("time_prefill", 0.0)
t_gen = final.get("time_generate", dt)
acc = final.get("accepted_draft_tokens", 0)
rej = final.get("rejected_draft_tokens", 0)

line = f"{n_gen} tokens in {t_gen:.2f}s — {n_gen / max(t_gen, 1e-9):.1f} tok/s"
if args.mtp or acc or rej:
    total = acc + rej
    pct = 100.0 * acc / total if total else 0.0
    line += f"  (MTP: {acc}/{total} drafts accepted, {pct:.0f}%)"
print(f"\x1b[2m{line}\x1b[0m", file=sys.stderr)

if args.timing:
    print(
        f"prefill: {n_prompt} tok in {t_prefill:.3f}s "
        f"({n_prompt / max(t_prefill, 1e-9):.1f} tok/s)  |  "
        f"decode: {n_gen} tok in {t_gen:.3f}s ({n_gen / max(t_gen, 1e-9):.1f} tok/s)  [mtp]"
        if args.mtp else
        f"prefill: {n_prompt} tok in {t_prefill:.3f}s "
        f"({n_prompt / max(t_prefill, 1e-9):.1f} tok/s)  |  "
        f"decode: {n_gen} tok in {t_gen:.3f}s ({n_gen / max(t_gen, 1e-9):.1f} tok/s)",
        file=sys.stderr,
    )
