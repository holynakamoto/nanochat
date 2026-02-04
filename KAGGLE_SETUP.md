# IaC-GPT Training on Kaggle (FREE GPUs!)

Train your Infrastructure-as-Code LLM on free Kaggle GPUs.

## ⚙️ Kaggle Setup

1. **Create a new notebook:** https://kaggle.com/code
2. **Enable GPU:** Settings → Accelerator → **GPU T4 x2**
3. **Enable Internet:** Settings → Internet → **On**
4. **Run all cells below**

---

## 🔧 Configuration (IMPORTANT!)

```python
# === CONFIGURATION ===
# T4 GPUs have 15GB VRAM - use smaller batch size!

MODEL_DEPTH = 12        # 12 (~300M, ~3hrs) | 16 (~500M, ~8hrs) | 20 (~800M, ~18hrs)
BATCH_SIZE = 4          # 4 for T4 GPUs (DO NOT increase to 8!)
NUM_GPUS = 2            # Kaggle T4 x2
WINDOW_PATTERN = "L"    # Use "L" (full attention) instead of "SSSL" for T4s

print(f"Training config: d{MODEL_DEPTH} model, batch_size={BATCH_SIZE}, gpus={NUM_GPUS}")
```

**Why batch_size=4?**
- T4 GPUs have 15GB VRAM (vs H100's 80GB)
- Without Flash Attention 3, memory usage is higher
- Batch size 8 = OOM error ❌
- Batch size 4 = Works perfectly ✅

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
    --eval-every=200 \
    --sample-every=500 \
    --save-every=1000"""

print(f"Running: {cmd}")
!{cmd}
```

**Key changes from original:**
- `--device-batch-size=4` (was 8)
- `--window-pattern=L` (was SSSL, which caused OOM)

**Training time:**
- d12: ~3 hours
- d16: ~8 hours
- d20: ~18 hours

---

## 📈 7. Evaluate Model

```bash
!python3 -m scripts.base_eval --device-batch-size=4
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
Batch size: 4
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

## 💡 Pro Tips

1. **Use Progressive Training**
   - Start with d12 (3hrs) to validate
   - Then try d16 (8hrs) or d20 (18hrs)

2. **Monitor GPU Usage**
   ```bash
   !nvidia-smi -l 1  # Live monitoring
   ```

3. **Save Checkpoints**
   - Model auto-saves every 1000 steps
   - Can resume if Kaggle disconnects

4. **Kaggle Limits**
   - 30 GPU hours/week (free tier)
   - 12 hour session limit
   - Plan your training accordingly!

---

## 📚 Next Steps

After downloading your model:

1. **Local Inference:** Use `scripts/chat_cli.py` or `scripts/chat_web.py`
2. **Test on Real IaC:** Generate Terraform/K8s configs
3. **Fine-tune:** Add your organization's IaC patterns
4. **Deploy:** Run as a local API for your dev team

---

## 🎉 You Did It!

You just trained a domain-specific LLM for Infrastructure-as-Code **for FREE** on Kaggle.

**Cost:** $0  
**Time:** 3-18 hours  
**Result:** Your own IaC expert model

Share your results with the community! 🚀

---

*Last updated: 2026-02-04*  
*Tested on: Kaggle T4 x2 GPUs*  
*Model: nanochat-based IaC-GPT*
