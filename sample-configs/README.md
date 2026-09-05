# Sample server configs

Working configs for the `server` binary, copied from a live tabbyAPI setup.
Pass one with `--config`:

```
./run-server.sh --config sample-configs/config-dflash.yml --host 0.0.0.0
```

`model_dir` and `draft_model_dir` are resolved **relative to the config file**,
not the process CWD — so a config living here looks for `sample-configs/models/`.
Point them at an absolute path, or keep the config next to your models.

| file | target | draft | max_seq_len |
|---|---|---|---|
| `config.example.yml` | — | — | annotated reference, every key documented |
| `config.yml` | heretic-ara 4.0bpw | mtp | 204800 |
| `config-dflash.yml` | heretic-ara 4.0bpw | dflash2 | 204800 |
| `config-dflash-180.yml` | heretic-ara 4.0bpw | dflash2 | 180224 |
| `config-dflash-35.yml` | base 3.5bpw | dflash2 | 180224 |
| `config-mtp-bench.yml` | heretic-ara 4.0bpw | mtp | 180224 |
| `config-no-vision.yml` | heretic-ara 4.0bpw | — | 204800 |

These ask for more context than the KV pool will fit on most cards. The server
sizes the pool to the VRAM actually left after the weights load and logs
`KV pool sized to N tokens (config asked for ...)`. That warning is normal;
`n_ctx` then advertises the real figure rather than the configured one.

`max_batch_size: 1` throughout: the Qwen3.5 hybrid keeps `draft_tokens + 1` GDN
history planes across 48 linear-attention layers **per slot**, so each extra
concurrent slot costs multiple GB that would otherwise be context.
