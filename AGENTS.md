# Repository development instructions

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->

## Architecture and validation

- Read DESIGN.md and CONTRIBUTING.md. Keep the main CLI lightweight and model imports inside independently launched runtime processes.
- Keep the installable package in src/locomanilab. Include runtime scripts and YAML resources in build artifacts; validate a wheel outside the source checkout.
- Treat workspace configuration, datasets, checkpoints, outputs, and user-provided repository paths as user-owned. Preserve source provenance and do not silently change controller/action contracts.
- Initialize only dependencies needed for the task. Their gitlinks are the revision source of truth. Do not switch to a floating upstream revision or run automatic checkout-cleaning/patching as installation behavior.
- Run appropriate CPU checks for changes. GPU training, hardware execution, large asset downloads, and optional SDK builds are separate operations, not prerequisites for ordinary development.
- Keep public recipes portable. Never commit credentials, concrete private host addresses, personal filesystem paths, experiment logs, weights, or private datasets.
- Report command/data checks separately from actual model and hardware validation. Do not describe an untested adapter as a supported end-to-end installation.
- The main project is MIT-licensed (`LICENSE`); do not apply that license to dependencies, weights, data, or assets. Run `python scripts/release/check_public.py` before committing; it flags common private paths and credentials.
