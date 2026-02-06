"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=49152, help='Vocabulary size (default: 49152 = 48K = 3×2^14). 48K for IaC-GPT: covers resource type identifiers, K8s API objects, Ansible modules, plus English text tokens.')
parser.add_argument('--validate', action='store_true', help='Run post-training validation suite with IaC-specific compression ratio tests')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# IaC-specific sanity check
iac_test = 'resource "aws_vpc" "main" {\n  cidr_block = var.vpc_cidr\n  enable_dns_hostnames = true\n}'
iac_encoded = tokenizer.encode(iac_test)
iac_decoded = tokenizer.decode(iac_encoded)
assert iac_decoded == iac_test

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# -----------------------------------------------------------------------------
# Optional IaC-GPT validation suite
if args.validate:
    print("\n" + "=" * 80)
    print("IaC-GPT Tokenizer Validation Suite")
    print("=" * 80)

    test_cases = [
        ("TF interpolation",   '${aws_instance.web.id}'),
        ("CIDR notation",      '10.0.0.0/16'),
        ("K8s resource",       'apiVersion: apps/v1\nkind: Deployment'),
        ("Ansible task",       '- name: Install Docker\n  apt:\n    name: docker-ce\n    state: present'),
        ("HCL variable block", 'variable "vpc_cidr" {\n  description = "CIDR block for VPC"\n  type = string\n  default = "10.0.0.0/16"\n}'),
        ("English requirement", 'Create an AWS VPC with public and private subnets across three availability zones'),
    ]

    print(f"\n{'Test Name':<25} | {'Chars':>6} | {'Tokens':>6} | {'Chars/Tok':>9}")
    print("-" * 55)

    for test_name, test_string in test_cases:
        enc_tokens = tokenizer.encode(test_string)
        n_chars = len(test_string)
        n_tokens = len(enc_tokens)
        ratio = n_chars / n_tokens if n_tokens > 0 else 0.0

        is_code = test_name != "English requirement"
        threshold = 4.0 if is_code else 5.0
        flag = "" if ratio >= threshold else " (low)"
        print(f"{test_name:<25} | {n_chars:>6} | {n_tokens:>6} | {ratio:>8.2f}{flag}")

    print("=" * 55)
    print("Target: >= 4.0 chars/token for IaC code, >= 5.0 for English")
    print()

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
