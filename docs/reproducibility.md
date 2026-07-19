# Reproducibility checklist

Complete this checklist before promoting a result as independently reproducible.

## Dataset and protocol

- Record the exact Amazon Reviews source, category, version, and checksum.
- Publish all filtering, ID remapping, split, and sampled-negative parameters.
- Preserve the evaluated user/item counts and exclusions for every run.
- State clearly whether evaluation is sampled-negative or full-catalog.

## Training

- Record the Git commit SHA, immutable config, Python and dependency versions.
- Record hardware, runtime, seed, batch size, stopping epoch, and checkpoint hash.
- Run at least three independent seeds for each learned model.

## Reporting

- Publish per-seed metrics, mean, standard deviation, and confidence intervals.
- Save raw prediction/rank files sufficient to recompute every table.
- Generate README tables from checked-in result files with a script.
- Separate synthetic smoke-test output from real-data performance results.

Suggested release layout:

```text
results/<release>/
├── manifest.json
├── environment.txt
├── dataset_checksums.txt
├── predictions/
├── runs/
└── tables/
```
