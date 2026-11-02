# Releasing a version

LocoManiLab is developed in this repository. A release is a tagged commit on `main`
with a GitHub Release; there is no separate export step.

## Prepare the release commit

1. Set the version in `pyproject.toml`, `src/locomanilab/__init__.py`, and the
   `locomanilab` entry of `uv.lock`, and update the version check in
   `tests/test_cli_workspace.py`.
2. Summarize user-visible changes under a new heading in [CHANGELOG.md](../CHANGELOG.md).
3. If dependencies changed, commit and push the dependency revision first, then commit
   the superproject's gitlink update. Every gitlink must point to a publicly
   accessible commit.

## Release gates

- CI passes on the release commit: lint, CPU tests, `scripts/release/check_public.py`,
  the policy-guide CPU examples, dependency plans, and the installed-wheel walkthrough.
- `scripts/release/check_public.py` reports no private paths or credentials. It checks
  a limited set of patterns; also review the diff since the previous release for host
  addresses, private experiment records, logs, weights, datasets, caches, and SDK binaries.
- The sdist and wheel contain `LICENSE`, and `pyproject.toml` declares `license = "MIT"`.
  MIT does not relicense dependencies, weights, data, or assets; review
  [third-party boundaries](THIRD_PARTY.md) when dependencies or released artifacts change.
- Documentation links, CLI examples, and the [model status](models.md) match what was
  validated. Record model, GPU, and hardware checks separately from CPU results.
- Released data, assets, and checkpoints referenced by the documentation are available in
  [`GammaRobotics/LocoManiLab`](https://huggingface.co/datasets/GammaRobotics/LocoManiLab).

## Publish

```bash
gh release create vX.Y.Z --target main --title "LocoManiLab vX.Y.Z" --notes-file <notes.md>
```

Never rewrite the history of `main` after it is published.

## Dependency version policy

Gitlinks are the dependency lock. Profiles only select which repositories are
initialized. Do not add a second hand-maintained set of divergent SHA pins.

Existing local environments, Git branches, unpushed changes, checkpoint directories, and
external `repos.*` paths must survive updates. Installation tools must not reset, clean,
or delete them. An unavailable dependency should fail with a clear access/version message
rather than silently cloning a different upstream commit.

## Validation record

Record the LocoManiLab revision, dependency revisions, Python/platform, executed CPU
commands, and any optional model environment checks. An adapter or manifest being present
is not end-to-end validation. Do not publish benchmark numbers without the datasets,
protocol, and artifacts required to reproduce them.
