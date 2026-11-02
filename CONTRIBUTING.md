# Contributing

Read [DESIGN.md](DESIGN.md) and [AGENTS.md](AGENTS.md) before changing execution boundaries. Start with a small, reproducible issue or pull request. Include the command, expected behavior, actual behavior, LocoManiLab revision, and selected dependency revisions; redact local paths, credentials, host addresses, and private data.

## CPU development

```bash
conda create -y -n locomanilab-dev python=3.10
conda activate locomanilab-dev
python -m pip install -e '.[dev,data]'
ruff check src/locomanilab tests scripts/setup
pytest -q
python -m build
```

A uv environment (`uv venv --python 3.10 .venv`, then `uv pip install -e '.[dev,data]'`) works as well; avoid the system Python. Install `build` separately if it is not included in your development environment. The CPU suite uses temporary datasets and fake dependency directories. GPU/model/hardware checks are optional and must skip clearly when their environments are absent. A successful CPU suite does not establish a model's training or deployment quality.

For packaging changes, install the built wheel into a fresh environment outside the repository, then repeat the README sample-data workflow. Editable installs can hide missing package resources.

## Adapter changes

- Keep model imports out of the main CLI. Run model code in its own process and environment.
- Implement shared evaluation through the adapter's server hook. Keep transport validation, episode reset, action ordering, and controller variant checks explicit.
- Put machine settings in workspace configuration. Keep recipes portable and use a local checkpoint path.
- Add meaningful CPU checks for command construction, resource packaging, or data contracts. For a model execution change, describe the environment and actual runtime validation separately.
- Preserve upstream license notices when moving integration code. A wrapper does not change a model's license.

## Dependency maintenance with Git

Initialize only the sources required for your work:

```bash
python3 scripts/setup/profiles.py act
python3 scripts/setup/profiles.py act --apply
git submodule status
```

A gitlink is the locked commit. The `branch` entry in `.gitmodules` is maintenance metadata; normal `git submodule update --init <path>` checks out the recorded commit, often in detached HEAD state. Prefix it with `GIT_LFS_SKIP_SMUDGE=1` as `scripts/setup/profiles.py` does: some dependencies (for example `external/lerobot`) track test fixtures in Git LFS that LocoManiLab does not use, and skipping the smudge avoids downloading them. Create a branch inside a dependency before developing there. Commit and make its revision available in the dependency repository, then review and commit the superproject's gitlink update. Never point a public release at an inaccessible commit.

Existing `env.local.yaml` `repos.*` overrides remain supported. They are user-owned checkouts: do not reset, delete, or silently relocate them. Inspect local changes before any dependency update. No custom cross-repository commit/push or cluster synchronization workflow is required.

## Release changes

Keep [CHANGELOG.md](CHANGELOG.md) concise. Contributions to the main project are accepted under its [MIT License](LICENSE). Open an issue or a pull request against `main`; CI runs the CPU checks, including `scripts/release/check_public.py` for private paths and credentials. Maintainers follow [docs/RELEASING.md](docs/RELEASING.md) to publish a version.
