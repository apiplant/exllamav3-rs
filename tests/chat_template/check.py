#!/usr/bin/env python3
"""Differential test: our hand-written ChatML renderer vs the model's real Jinja.

`src/server/chat.rs` transcribes `chat_template.jinja` by hand (this port has no
Jinja engine). Transcription drifts, and drift is not cosmetic — it puts the
model out of distribution and it starts leaking foreign tool-call syntax into
content. This renders the same fixtures both ways and requires them byte-equal.

    python crate/tests/chat_template/check.py [MODEL_DIR]

Needs `jinja2` and a built test binary; exits non-zero on any difference.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
CRATE = HERE.parent.parent
DEFAULT_MODEL = "/mnt/extra/ai/llm/qwen3.8-27B/Qwen3.8-27B-heretic-ara-exl3-4.0bpw"


def render_reference(template_path, cases):
    import jinja2

    def raise_exception(msg):
        raise RuntimeError(msg)

    env = jinja2.Environment()
    env.globals["raise_exception"] = raise_exception
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": False}
    tpl = env.from_string(template_path.read_text())
    return [tpl.render(add_generation_prompt=True, **c) for c in cases]


def render_ours(cases_path, out_path):
    env = dict(
        os.environ,
        LIBTORCH_USE_PYTORCH="1",
        LIBTORCH_BYPASS_VERSION_CHECK="1",
        TORCH_CUDA_ARCH_LIST="8.9",
        EXL3_TPL_CASES=str(cases_path),
        EXL3_TPL_OUT=str(out_path),
    )
    subprocess.run(
        ["cargo", "test", "--release", "dump_template_fixtures"],
        cwd=CRATE, env=env, check=True, stdout=subprocess.DEVNULL,
    )
    return json.loads(out_path.read_text())


def main():
    model = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    template = model / "chat_template.jinja"
    if not template.exists():
        sys.exit(f"no chat_template.jinja under {model}")

    cases_path = HERE / "cases.json"
    cases = json.loads(cases_path.read_text())
    ref = render_reference(template, cases)
    with tempfile.TemporaryDirectory() as td:
        ours = render_ours(cases_path, pathlib.Path(td) / "rust_out.json")

    import difflib

    bad = 0
    for i, (r, g) in enumerate(zip(ref, ours)):
        if r != g:
            bad += 1
            print(f"===== case {i} differs =====")
            print("\n".join(difflib.unified_diff(
                r.splitlines(), g.splitlines(), "jinja", "rust", lineterm="", n=1)))
    print(f"cases: {len(ref)}  differing: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
