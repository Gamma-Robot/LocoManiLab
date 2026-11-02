# Domain randomization and evaluation settings

Treat scene/task initialization, robot reset jitter, visual changes, and action noise as separate experimental settings. They should not silently change between collection and evaluation.

Inspect `lab eval --help` for supported options. Record the selected scene, reset distribution, seed, controller checkpoint, action horizon, and policy query rate with each result. Keep nominal and perturbed runs separate, and report the perturbation definition with any comparison.

A fixed scene does not by itself establish diverse independent resets. A repeated seed can support reproducibility, but it is not evidence that task behavior generalizes. Verify task initialization in the pinned simulator before interpreting success rates. No published benchmark numbers are carried forward from private experiment records.
