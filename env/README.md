# Free-tier runners

The whole project runs on free compute. The **OT merge is CPU** (seconds); the GPU is only for the
8-task evaluation.

## Kaggle (recommended — 30 GPU-hr/week, T4/P100)

1. New Notebook → Settings → Accelerator: **GPU T4 x2** (or P100).
2. First cell:
   ```bash
   !pip install -q POT==0.9.6 peft transformers open_clip_torch
   !pip install -q git+https://github.com/tanganke/fusion_bench.git
   !git clone <this-repo> && cd ot-lora-merge && pip install -e .
   ```
3. Run a milestone:
   ```bash
   !cd ot-lora-merge && python experiments/m0_baselines/run.py --config configs/modelpool/clip_vit_b32_knots_8task_lora.yaml
   !cd ot-lora-merge && python experiments/m1_ot_v1/run.py   --config configs/modelpool/clip_vit_b32_knots_8task_lora.yaml --method configs/method/ot_lora_barycenter.yaml
   ```

## Colab (free, overflow)

Same install; use for the CPU merge + small evals when the Kaggle weekly budget is spent.

## CPU-only (no accelerator) — works today

```bash
pip install -e . && pytest -q            # the OT core tests
python examples/demo_synthetic.py        # end-to-end merge on synthetic adapters
python experiments/m3_carveout/run.py --timing-only   # merge-time benchmark (the efficiency result)
```

## Budget (Spec §5)

| Milestone | GPU-hr est. | Fits free tier? |
|---|---|---|
| M0 baselines (+ optional specialist training) | ≤ ~20 | ✅ |
| M1 OT v1 | ~3 | ✅ |
| M2 ablations | ~10–15 | ✅ |
| M3 carve-out | ~3 | ✅ |
| M4 write-up reruns | ~3 | ✅ |

⚠ Stay out of v1: ViT-L/14 or Llama-3-8B full eval (can blow the weekly free budget), paid wandb,
gated models. Defer scale-up to v2 once the method is proven on ViT-B/32.

## Kaggle MCP (agent-driven runs)

The **official Kaggle MCP server** is registered globally in Claude Code (user scope), so any agent
can drive Kaggle directly — no manual notebook upload needed.

- **Endpoint:** `https://www.kaggle.com/mcp` (remote HTTP, OAuth 2.0, first-party, zero local install).
- **Registered with:** `claude mcp add --transport http kaggle https://www.kaggle.com/mcp --scope user`
  (config in `~/.claude.json`; remove with `claude mcp remove kaggle -s user`).
- **Authenticate (one-time, user action):** in a Claude Code session run `/mcp`, select `kaggle`, and
  complete the browser OAuth login to your Kaggle account. Tools needing auth stay locked until then.
- **Tools:** ~57 across competitions, **notebooks/kernels**, datasets, models, benchmarks, forums.

**Intended workflow for this project** (once authenticated): use the MCP's kernel tools to push
`env/kaggle-notebook.ipynb` to Kaggle, run it on the free GPU, and pull results back into `results/` —
replacing the manual upload above. The manual-Kaggle and CPU-only paths remain as fallbacks.

> Note: newly added MCP servers load at the **start** of a Claude Code session, so the `kaggle` tools
> appear after restarting/reloading the session.
