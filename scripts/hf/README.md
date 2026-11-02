# Hugging Face release tools

| Tool | Purpose |
|---|---|
| `download.py` | List and download benchmark data, the simulator assets selected tasks need (per file, from their original hosts, sha256-verified), and released checkpoints |
| `upstream.py` | Per-file fetchers used by `download.py assets`: remote zip members (HTTP ranges), MolmoSpaces packages, plain files |
| `minimal_test.py` | Minimal download-and-test unit: `cpu`, `checkpoint`, `assets`, and `gpu` stages with PASS/FAIL per stage |
| `gr00t_openloop.py` | Compare a running GR00T server's actions with recorded actions on benchmark frames |

Install the main package with `pip install -e '.[data,hf]'` and run the tools from the
checkout root. See [data, assets, and checkpoints](../../docs/hf_resources.md) for the
repository layout, commands, and what each stage does and does not validate.
