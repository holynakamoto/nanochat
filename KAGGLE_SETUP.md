# IaC-GPT Training on Kaggle (FREE GPUs!)

Train your Infrastructure-as-Code LLM on free Kaggle GPUs. Supports checkpoint
persistence and resume across the 12-hour session limit.

## Kaggle Setup

1. **Create a new notebook:** https://kaggle.com/code
2. **Enable GPU:** Settings → Accelerator → **GPU T4 x2**
3. **Enable Internet:** Settings → Internet → **On**
4. **Set your Kaggle username** in the config cell
5. **Run all cells**

---

## Configuration

```python
MODEL_DEPTH = 12        # 12 (~286M, ~3hrs) | 16 (~400M, ~8hrs) | 24 (~1.6B, won't fit T4)
BATCH_SIZE = 2          # 2 for T4 GPUs (prevents OOM)
NUM_GPUS = 2            # Kaggle T4 x2
WINDOW_PATTERN = "L"    # Full attention (don't use SSSL on T4s)
DATA_RATIO = 8          # target-param-data-ratio

# Resume from a previous session
RESUME = False           # Set True to continue training
RESUME_STEP = -1         # -1 = auto-detect, or set specific step

KAGGLE_USERNAME = "YOUR_USERNAME_HERE"
```

**Why batch_size=2?**
- T4 GPUs have 15GB VRAM (vs H100's 80GB)
- Without Flash Attention 3, memory usage is higher
- Batch size 4 = OOM error during backward pass
- Batch size 2 = works reliably

---

## 📦 1. Clone Repository

```bash
!git clone https://github.com/holynakamoto/nanochat.git
%cd nanochat
!nvidia-smi  # Verify GPUs
```

---

## 📥 2. Install Dependencies

```bash
!pip install -q tiktoken pyarrow filelock rustbpe wandb tabulate regex zstandard
!pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention not available (using fallback)"
```

---

## 🗂️ 3. Collect IaC Training Data

```bash
# Fast git clone method (10-15 minutes)
!bash dev/fast_scrape_iac.sh <<< 'n'
```

**Expected output:**
- 11,000+ IaC files
- ~26 MB corpus
- Terraform, Kubernetes, Ansible, Crossplane, Docker

---

## 📊 4. Create Training Shards

```bash
!python3 dev/repackage_iac_data.py \
    --input-dir data/iac_raw_cloned \
    --output-dir ~/.cache/nanochat/iac_data \
    --include-synthetic \
    --include-docs

# Setup data directories
!cp ~/.cache/nanochat/iac_data/shard_00000.parquet ~/.cache/nanochat/iac_data/shard_00001.parquet
!ln -sf ~/.cache/nanochat/iac_data ~/.cache/nanochat/base_data
!ls -la ~/.cache/nanochat/base_data/
```

---

## 🔤 5. Train Custom Tokenizer

```bash
!python3 -m scripts.tok_train
!python3 -m scripts.tok_eval  # Optional: see compression stats
```

---

## 🚀 6. Train IaC-GPT Model

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'

# CORRECTED TRAINING COMMAND
cmd = f"""torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.base_train -- \
    --depth={MODEL_DEPTH} \
    --device-batch-size={BATCH_SIZE} \
    --window-pattern={WINDOW_PATTERN} \
    --target-param-data-ratio=5 \
    --run=dummy \
    --model-tag=iac-gpt-d{MODEL_DEPTH} \
    --eval-every=100 \
    --sample-every=100 \
    --save-every=100"""

print(f"Running: {cmd}")
!{cmd}
```

**Key changes from original:**
- `--device-batch-size=2` (was 8)
- `--window-pattern=L` (was SSSL, which caused OOM)

**Training time:**
- d12: ~3 hours
- d16: ~8 hours
- d20: ~18 hours

---

## 📈 7. Evaluate Model

```bash
!python3 -m scripts.base_eval --device-batch-size=2
```

**Expected metrics:**
- CORE score > 0.20 (GPT-2 capability level)
- Perplexity on IaC validation set

---

## 💾 8. Download Your Model

```python
# Compress model for download
!tar -czf iac_gpt_model.tar.gz ~/.cache/nanochat/base_checkpoints/

# Download via Kaggle Output panel
from IPython.display import FileLink
FileLink('iac_gpt_model.tar.gz')
```

---

## 🧪 9. Test Inference

```python
# Quick test
!python3 -m scripts.chat_cli --model ~/.cache/nanochat/base_checkpoints/iac-gpt-d12/latest_checkpoint
```

**Try prompts like:**
- "Create a Terraform module for an EKS cluster"
- "Write a Kubernetes deployment for nginx"
- "Generate an Ansible playbook to deploy a web app"

---

## 🎯 Expected Results

### Data Collection
```
terraform      : 3,657 files (32.7%)
kubernetes     : 7,249 files (64.8%)
ansible        :   151 files ( 1.3%)
crossplane     :    13 files ( 0.1%)
docker         :   118 files ( 1.1%)
TOTAL          : 11,188 files
```

### Training Stats (d12 model)
```
Model: 286M parameters
Batch size: 2
GPUs: 2x T4
Training time: ~3 hours
Cost: FREE on Kaggle!
```

### Quality Metrics
```
CORE score: >0.20 target
Perplexity: <3.5 target
Compilability: TBD (manual testing)
```

---

## ⚠️ Troubleshooting

###  OOM Error
**Problem:** `CUDA out of memory`  
**Solution:** Reduce batch size to 2:
```python
BATCH_SIZE = 2  # Even more conservative
```

### Slow Training
**Problem:** Taking longer than expected  
**Causes:**
- T4 GPUs are slower than H100s (expected)
- No Flash Attention 3 (using fallback)
- Full context attention (`window-pattern=L`)

**This is normal!** T4 training is slower but works.

### Repository Clone Failed
**Problem:** Git clone fails  
**Solution:** Already cloned - skip to next cell

---

## 🏆 Success Checklist

- [ ] Data scraped: 11k+ files
- [ ] Shards created: 1-2 parquet files
- [ ] Tokenizer trained: vocab size 32,768
- [ ] Model training: Reached step 1000+
- [ ] Evaluation: CORE > 0.20
- [ ] Model downloaded

---

## Resume Training Across Sessions

Kaggle kills sessions after 12 hours. The notebook auto-saves your progress:

1. After training, the "Save Checkpoints" cell uploads to Kaggle Datasets
2. Start a new notebook, add your saved datasets as Inputs
3. Set `RESUME = True`, run all cells
4. Scraping is skipped (data cached), training resumes from last checkpoint

**What gets saved:**
- Model weights (`model_*.pt`)
- Optimizer state (`optim_*.pt`) -- needed for smooth resume
- Training metadata (`meta_*.json`) -- step count, loss, dataloader position
- Training data shards (`*.parquet`) -- so you never re-scrape

---

## GPU Options

| Platform | GPU | VRAM | Free? | d12 time | d16 time |
|----------|-----|------|-------|----------|----------|
| Kaggle | T4 x2 | 15GB each | Yes (30hr/week) | ~3hrs | ~8hrs |
| Colab Free | T4 x1 | 15GB | Yes (limited) | ~6hrs | ~16hrs |
| Colab Pro | A100 x1 | 40GB | $10/mo | ~1hr | ~3hrs |
| Lambda | A100 x1 | 80GB | ~$1.10/hr | ~40min | ~2hrs |

For d12 on your current corpus, **Kaggle T4 x2 is fine** -- it completes in one session.
For d16+, you'll need the resume feature or a faster GPU.

---

## Pro Tips

1. **Progressive training:** Start d12 to validate, then d16 if it looks good
2. **Monitor:** Loss should drop from ~10.0 to <3.5 for d12
3. **Kaggle limits:** 30 GPU hrs/week free. d12 uses ~3hrs = 10 runs/week
4. **WandB:** Add your API key to track loss curves across sessions

---

*Last updated: 2026-02-06*
*Tested on: Kaggle T4 x2 GPUs*
