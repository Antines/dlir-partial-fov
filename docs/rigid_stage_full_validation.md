# Full 1,080-Slab Rigid Validation

The method and hyperparameters are frozen from the earlier 36-case validation screen. The 1,080 saved slabs in `dlir_data/test/` are the validation split from 30 subjects; the separate held-out final test set is not used.

Evaluate only header-initialized support-aware negative-NCC rigid refinement. Ground truth is used for reporting TRE, rigid oracle and similarity diagnostics; it is not used by the optimizer to select a transform. It is a numerical registration baseline, **not** an end-to-end trained deep-learning result.

Resume is per case through `cases.jsonl`, with each optimized transform saved in `transforms/`. A case is skipped only if its result and transform both exist. W&B stays online in project `neuroml-dlir`, with persisted run IDs.

Failure is the established evaluator definition: rotation error greater than 2 degrees **or** centroid translation error greater than 2 mm. `TRE < 2 mm` is a separate metric; tool failures are counted as failures.

Review mean TRE per case, median/p90 over cases, per-subgroup performance, 2mm rate, failures and worst outliers. Do not compare runtime to AFNI without matching hardware/preprocessing. Do not represent validation results as a held-out-test estimate.
