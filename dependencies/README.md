# Optional source profiles

`profiles.json` selects repositories; **the tracked gitlinks are the revision source of truth**. `.gitmodules` supplies URLs. No separate SHA list is maintained.

Twelve of the 14 current dependency URLs target repositories in the [LocoManiLab organization](https://github.com/LocoManiLab), including the renamed `humanoid_benchmark`; each keeps its upstream history and adds one LocoManiLab commit. `external/lerobot` uses upstream [Hugging Face LeRobot](https://github.com/huggingface/lerobot) at v0.3.3 and `external/opendm` uses upstream [Dexmal/OpenDM](https://github.com/Dexmal/OpenDM); LocoManiLab modifies neither. Verify repository access and availability of the recorded commit before initialization; an organization URL alone does not establish public access or installation readiness. Forking or copying code does not change its third-party license or weight/asset terms.

```bash
python3 scripts/setup/profiles.py --list
python3 scripts/setup/profiles.py core
python3 scripts/setup/profiles.py sim
python3 scripts/setup/profiles.py sim --apply
```

The default is a plan. `--apply` initializes missing selected submodules with Git LFS smudging disabled. It does not change initialized checkouts or provision environments, SDKs, weights, or assets. Fork access may be required. Review URLs and locked commits before applying a plan.

| Profile | Source dependency scope |
|---|---|
| `core`, `data` | None |
| `sim` | humanoid_benchmark + g1_backend |
| `real` | g1_backend |
| Policy ID (`act`, `dp`, `dm05`, etc.) | That policy's model implementation |
| `gr00t-real` | Isaac-GR00T + complete GR00T WholeBodyControl |
| `fastwam-real` | FastWAM + Isaac-GR00T client/runtime + complete GR00T WholeBodyControl |
| `wbc` | Native GR00T WholeBodyControl source for complete robot deployment |

Simulation and real-robot collection both use `g1_backend`; `sim` adds the simulation host and `real` selects the hardware runtime source. The `heft` controller is included in `g1_backend`, so no separate HeFT source profile is needed. Existing HeFT datasets and model variants retain their documented contracts.

For complete GR00T/FastWAM robot deployment, the native WholeBodyControl runtime selected by their `*-real` profiles is this project's recommended best practice.

Start with [minimal simulation setup](../docs/simulation.md) for EmptyEnv and selected controller weights; the source profile does not require a kitchen or iTHOR asset pack.

For simulation evaluation, select both the policy profile and `sim`. A code profile is not a tested environment lock. See the [model support matrix](../docs/models.md).

Use standard Git to inspect/update revisions. The setup helper never follows `branch = ...` automatically or replaces a fork with an upstream fallback. User-supplied workspace `repos.*` paths remain independent of source initialization.
