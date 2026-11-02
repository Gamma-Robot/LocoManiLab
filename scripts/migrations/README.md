# Explicit dataset maintenance

`gen_episodes_stats.py <dataset>` computes per-episode numeric statistics from LeRobot-style parquet data. It skips an existing output unless `--force` is supplied. Back up or copy a dataset before overwriting its metadata, and validate downstream normalization against that model's loader.

Historical controller/schema repair scripts and experiment-specific dataset transformations are not supported migration APIs in this release. Preserve the source and use a reviewed migration or recollect when action/state semantics differ. The ordinary `lab prepare` command is the public preparation entry point.
