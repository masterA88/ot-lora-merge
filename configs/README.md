# Configs

FusionBench-compatible YAML. Three groups:

- `method/` — OT-LoRA-Merge variants (align / barycenter / gw). These are the novel methods.
- `modelpool/` — the 8-task CLIP-ViT-B/32 LoRA pool (datasets + base model; all free).
- `baseline/` + `ablation/` — to be added in M0/M2; baselines reuse FusionBench's built-ins
  (task_arithmetic, ties, adamerging, knots, core_space), so comparison is apples-to-apples.

Method config keys map 1:1 to `ot_lora_merge.merge.merge_layer(**kwargs)`.
