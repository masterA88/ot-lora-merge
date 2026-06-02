"""Kaggle probe — resolves the VERIFY-IN-KAGGLE unknowns in one cheap run (no GPU needed).

Pushed + run on Kaggle via the Kaggle MCP (save_notebook, kernelType=script, internet ON, GPU OFF).
Checks, each isolated so one failure doesn't abort the rest:
  1. FusionBench installs + the exact API names DEA wired actually import.
  2. FusionBench's OWN bundled TA8-LoRA modelpool config (authoritative adapter repo IDs + rank).
  3. A tanganke CLIP LoRA adapter loads from HF, and we print its adapter_config (rank) + PEFT keys.
Output is printed to stdout (captured in the kernel log) and also written to /kaggle/working/probe_report.txt.
"""
import json, sys, traceback, importlib, glob, os

REPORT = []
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    REPORT.append(line)

def section(name):
    log("\n" + "=" * 8 + f" {name} " + "=" * 8)

# ---------------------------------------------------------------- install
section("INSTALL")
os.system(f"{sys.executable} -m pip install -q 'POT==0.9.6' peft open_clip_torch "
          "git+https://github.com/tanganke/fusion_bench.git 2>&1 | tail -n 5")

# ---------------------------------------------------------------- 1. API imports
section("1. FusionBench API imports (the names DEA wired)")
API = [
    ("fusion_bench.method.base_algorithm", "BaseAlgorithm"),
    ("fusion_bench.modelpool", "CLIPVisionModelPool"),
    ("fusion_bench.modelpool.base_pool", "BaseModelPool"),
    ("fusion_bench.taskpool.clip_vision", "CLIPVisionModelTaskPool"),
    ("fusion_bench.models.linearized.vision_model", "load_lora_vision_model_hf"),
    ("fusion_bench.models.linearized.vision_model", "load_fft_vision_model_hf"),
    ("fusion_bench.method.simple_average", "SimpleAverageAlgorithm"),
    ("fusion_bench.method.task_arithmetic", "TaskArithmeticAlgorithm"),
    ("fusion_bench.method.ties_merging", "TiesMergingAlgorithm"),
]
try:
    import fusion_bench
    log("fusion_bench version:", getattr(fusion_bench, "__version__", "?"))
except Exception as e:
    log("FATAL: cannot import fusion_bench:", e)
for mod, name in API:
    try:
        m = importlib.import_module(mod)
        ok = hasattr(m, name)
        log(f"  [{'OK ' if ok else 'MISSING'}] {mod}.{name}")
    except Exception as e:
        log(f"  [ERR] {mod}.{name} -> {type(e).__name__}: {e}")

# ---------------------------------------------------------------- 2. bundled config (authoritative repo IDs)
section("2. FusionBench bundled TA8-LoRA modelpool config (authoritative adapter IDs + rank)")
try:
    import fusion_bench as fb
    root = os.path.dirname(fb.__file__)
    cfg_root = os.path.join(os.path.dirname(root), "fusion_bench_config")
    hits = []
    for base in (root, cfg_root, os.path.dirname(root)):
        hits += glob.glob(os.path.join(base, "**", "*lora*.yaml"), recursive=True)
    hits = sorted(set(h for h in hits if "clip" in h.lower() or "modelpool" in h.lower()))
    log("candidate config files:")
    for h in hits[:20]:
        log("   ", h)
    # print the one that looks like the 8-task vit-b32 lora pool
    for h in hits:
        low = h.lower()
        if "lora" in low and ("ta8" in low or "eight" in low or "base-patch32" in low or "vit-b" in low):
            log(f"\n----- {h} -----")
            with open(h) as f:
                log(f.read())
            break
except Exception:
    log(traceback.format_exc())

# ---------------------------------------------------------------- 3. load one adapter from HF
section("3. Load a tanganke CLIP-ViT-B/32 LoRA adapter from HF (existence + rank + PEFT keys)")
CANDIDATES = [
    "tanganke/clip-vit-base-patch32_sun397_lora-16",
    "tanganke/clip-vit-base-patch32_sun397",
    "tanganke/clip-vit-base-patch32_sun397_lora",
]
from huggingface_hub import HfApi
api = HfApi()
found = None
for repo in CANDIDATES:
    try:
        info = api.model_info(repo, files_metadata=False)
        files = [s.rfilename for s in info.siblings]
        log(f"  [EXISTS] {repo}  files={files}")
        if found is None:
            found = repo
    except Exception as e:
        log(f"  [NOT FOUND] {repo} -> {type(e).__name__}: {str(e)[:120]}")

if found:
    try:
        from huggingface_hub import hf_hub_download
        cfgp = hf_hub_download(found, "adapter_config.json")
        cfg = json.load(open(cfgp))
        log("\n  adapter_config.json:")
        for k in ("r", "lora_alpha", "target_modules", "peft_type", "base_model_name_or_path"):
            log(f"    {k} = {cfg.get(k)}")
        # load as PeftModel onto CLIP vision and print a few lora keys
        import torch
        from transformers import CLIPVisionModel
        from peft import PeftModel
        base = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        pm = PeftModel.from_pretrained(base, found)
        lk = [k for k in pm.state_dict() if "lora_A" in k or "lora_B" in k]
        log(f"\n  PeftModel loaded. #lora keys = {len(lk)}. sample keys:")
        for k in lk[:4]:
            log("    ", k)
        log("  peft_config keys:", list(pm.peft_config.keys()))
        c0 = list(pm.peft_config.values())[0]
        log("  scale alpha/r =", c0.lora_alpha, "/", c0.r, "=", c0.lora_alpha / c0.r)
    except Exception:
        log(traceback.format_exc())
else:
    log("  No candidate adapter repo found — need to locate the correct HF IDs.")

# ---------------------------------------------------------------- write report
with open("/kaggle/working/probe_report.txt", "w") as f:
    f.write("\n".join(REPORT))
log("\n=== probe complete; report written to /kaggle/working/probe_report.txt ===")
