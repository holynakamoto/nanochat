"""
Sanitize IaC training data to remove secrets and high-entropy strings.

This script scans IaC training data (parquet shards or raw files) for secrets
and sensitive information that should NOT be baked into model weights. It detects
and redacts:
- AWS Access Keys and Secret Keys
- SSH private keys
- GCP service account keys
- API tokens and passwords
- Real public IP addresses (keeps RFC1918 private ranges)
- Base64-encoded blobs (likely certificates/keys)

The script is conservative: better to redact a false positive than miss a real secret.

Usage:
    # Scan and sanitize parquet shards (dry-run)
    python dev/sanitize_iac.py --dry-run

    # Actually sanitize shards in place
    python dev/sanitize_iac.py

    # Sanitize with different input/output
    python dev/sanitize_iac.py --input-dir ~/.cache/nanochat/iac_data --output-dir ~/.cache/nanochat/iac_data_clean

    # Also scan raw source files
    python dev/sanitize_iac.py --raw-dir data/iac_raw --dry-run
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import pyarrow.parquet as pq
import pyarrow as pa


# =============================================================================
# Secret detection patterns
# =============================================================================

PATTERNS = {
    'aws_access_key': re.compile(r'AKIA[0-9A-Z]{16}'),
    'aws_secret_key': re.compile(r'(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])'),
    'ssh_private_key': re.compile(r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'ssh_private_key_block': re.compile(
        r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (RSA |EC |OPENSSH )?PRIVATE KEY-----',
        re.DOTALL
    ),
    'gcp_service_key': re.compile(r'"private_key":\s*"-----BEGIN'),
    'generic_api_key': re.compile(
        r'(?:api[_-]?key|token|password|secret|auth[_-]?key|access[_-]?token)'
        r'["\']?\s*[:=]\s*["\']([A-Za-z0-9_\-]{20,})["\']',
        re.IGNORECASE
    ),
    'base64_blob': re.compile(r'(?<![A-Za-z0-9/+=])[A-Za-z0-9+/]{64,}={0,2}(?![A-Za-z0-9/+=])'),
}

# IP address pattern
IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

# RFC1918 private ranges and common safe IPs
SAFE_IP_PREFIXES = [
    '10.',
    '172.16.', '172.17.', '172.18.', '172.19.',
    '172.20.', '172.21.', '172.22.', '172.23.',
    '172.24.', '172.25.', '172.26.', '172.27.',
    '172.28.', '172.29.', '172.30.', '172.31.',
    '192.168.',
    '127.',
    '0.0.0.',
    '255.255.255.',
    '169.254.',  # Link-local
    '192.0.2.',  # RFC5737 TEST-NET-1
    '198.51.100.',  # RFC5737 TEST-NET-2
    '203.0.113.',  # RFC5737 TEST-NET-3
]

# Replacement placeholders
REPLACEMENTS = {
    'aws_access_key': 'AKIAIOSFODNN7EXAMPLE',
    'aws_secret_key': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
    'ssh_private_key': '<REDACTED_PRIVATE_KEY>',
    'gcp_service_key': '"private_key": "<REDACTED_GCP_PRIVATE_KEY>',
    'generic_api_key': '<REDACTED_TOKEN>',
    'base64_blob': '<REDACTED_BASE64>',
    'real_ip': '203.0.113.X',
}


# =============================================================================
# Helper functions
# =============================================================================

def is_safe_ip(ip: str) -> bool:
    """Check if an IP address is safe (private/example/documentation ranges)."""
    for prefix in SAFE_IP_PREFIXES:
        if ip.startswith(prefix):
            return True
    return False


def is_likely_version_or_hash(text: str, match_obj) -> bool:
    """
    Check if a matched 40-char pattern is likely a git SHA or version string
    rather than an AWS secret key.
    """
    matched_text = match_obj.group(0)

    # Git SHAs are hex-only
    if re.match(r'^[a-f0-9]{40}$', matched_text, re.IGNORECASE):
        return True

    # Check surrounding context for version/hash keywords
    start = max(0, match_obj.start() - 30)
    end = min(len(text), match_obj.end() + 30)
    context = text[start:end].lower()

    version_keywords = ['version', 'commit', 'sha', 'hash', 'digest',
                        'checksum', 'sha256', 'sha1', 'md5', 'image:']
    if any(kw in context for kw in version_keywords):
        return True

    return False


def sanitize_text(text: str, stats: Dict[str, int]) -> str:
    """
    Sanitize a single text document by redacting secrets.
    Updates stats dictionary with counts of each secret type found.
    """
    sanitized = text

    # 1. AWS Access Keys
    for match in PATTERNS['aws_access_key'].finditer(sanitized):
        sanitized = sanitized.replace(match.group(0), REPLACEMENTS['aws_access_key'])
        stats['aws_access_key'] += 1

    # 2. AWS Secret Keys (careful with false positives)
    for match in PATTERNS['aws_secret_key'].finditer(text):
        if not is_likely_version_or_hash(text, match):
            matched = match.group(0)
            if matched in sanitized:
                sanitized = sanitized.replace(matched, REPLACEMENTS['aws_secret_key'])
                stats['aws_secret_key'] += 1

    # 3. SSH Private Keys (replace full block)
    if PATTERNS['ssh_private_key'].search(sanitized):
        sanitized = PATTERNS['ssh_private_key_block'].sub(
            REPLACEMENTS['ssh_private_key'], sanitized
        )
        stats['ssh_private_key'] += 1

    # 4. GCP Service Account Keys
    if PATTERNS['gcp_service_key'].search(sanitized):
        sanitized = PATTERNS['gcp_service_key'].sub(
            REPLACEMENTS['gcp_service_key'], sanitized
        )
        stats['gcp_service_key'] += 1

    # 5. Generic API Keys/Tokens (replace just the value)
    for match in PATTERNS['generic_api_key'].finditer(text):
        token = match.group(1)
        if token in sanitized:
            sanitized = sanitized.replace(token, REPLACEMENTS['generic_api_key'])
            stats['generic_api_key'] += 1

    # 6. Base64 blobs in key-value context
    for match in PATTERNS['base64_blob'].finditer(text):
        matched_text = match.group(0)
        start = max(0, match.start() - 15)
        context_before = text[start:match.start()]
        if any(c in context_before for c in ['=', ':', '"', "'"]):
            if matched_text in sanitized:
                sanitized = sanitized.replace(matched_text, REPLACEMENTS['base64_blob'])
                stats['base64_blob'] += 1

    # 7. Real IP addresses (keep private/documentation ranges)
    for match in IP_PATTERN.finditer(text):
        ip = match.group(0)
        if not is_safe_ip(ip):
            if ip in sanitized:
                sanitized = sanitized.replace(ip, REPLACEMENTS['real_ip'])
                stats['real_ip'] += 1

    return sanitized


# =============================================================================
# Processing functions
# =============================================================================

def sanitize_parquet_shards(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Sanitize all parquet shards in input_dir.
    Returns (stats, affected_files) dicts.
    """
    stats = defaultdict(int)
    affected_files = defaultdict(list)

    if not input_dir.exists():
        print(f"WARNING: Input directory {input_dir} does not exist")
        return stats, affected_files

    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    if not shard_files:
        # Also try *.parquet in case naming differs
        shard_files = sorted(input_dir.glob("*.parquet"))

    if not shard_files:
        print(f"WARNING: No parquet shards found in {input_dir}")
        return stats, affected_files

    print(f"\nScanning {len(shard_files)} parquet shards...")
    print("=" * 60)

    for shard_path in shard_files:
        shard_stats = defaultdict(int)

        table = pq.read_table(shard_path)
        texts = table['text'].to_pylist()

        sanitized_texts = []
        for text in texts:
            if text is None or text == "":
                sanitized_texts.append(text)
                continue
            sanitized = sanitize_text(text, shard_stats)
            sanitized_texts.append(sanitized)

        if any(count > 0 for count in shard_stats.values()):
            for secret_type, count in shard_stats.items():
                if count > 0:
                    affected_files[secret_type].append(shard_path.name)
                    stats[secret_type] += count

            print(f"{shard_path.name}: {dict(shard_stats)}")

            if not dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / shard_path.name

                sanitized_table = pa.Table.from_pydict({"text": sanitized_texts})
                pq.write_table(
                    sanitized_table,
                    str(output_path),
                    row_group_size=1024,
                    use_dictionary=False,
                    compression="zstd",
                    compression_level=3,
                    write_statistics=False,
                )
        else:
            print(f"{shard_path.name}: clean")

    return stats, affected_files


def sanitize_raw_files(
    raw_dir: Path,
    dry_run: bool = False
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Sanitize raw IaC files (.tf, .yaml, .yml, .hcl) in place.
    Returns (stats, affected_files) dicts.
    """
    stats = defaultdict(int)
    affected_files = defaultdict(list)

    if not raw_dir.exists():
        print(f"WARNING: Raw directory {raw_dir} does not exist")
        return stats, affected_files

    extensions = ["**/*.tf", "**/*.yaml", "**/*.yml", "**/*.hcl", "**/*.tfvars"]
    raw_files = []
    for pattern in extensions:
        raw_files.extend(raw_dir.glob(pattern))

    if not raw_files:
        print(f"WARNING: No raw IaC files found in {raw_dir}")
        return stats, affected_files

    print(f"\nScanning {len(raw_files)} raw files...")
    print("=" * 60)

    for file_path in raw_files:
        file_stats = defaultdict(int)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        sanitized = sanitize_text(text, file_stats)

        if any(count > 0 for count in file_stats.values()):
            rel = file_path.relative_to(raw_dir)
            for secret_type, count in file_stats.items():
                if count > 0:
                    affected_files[secret_type].append(str(rel))
                    stats[secret_type] += count

            print(f"{rel}: {dict(file_stats)}")

            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(sanitized)

    return stats, affected_files


# =============================================================================
# Reporting
# =============================================================================

def print_report(
    stats: Dict[str, int],
    affected_files: Dict[str, List[str]],
    dry_run: bool
):
    """Print a summary report of sanitization results."""
    print("\n" + "=" * 60)
    print("SANITIZATION REPORT")
    print("=" * 60)

    if dry_run:
        print("MODE: DRY-RUN (no files were modified)")
    else:
        print("MODE: LIVE (files have been sanitized)")
    print()

    if not stats:
        print("No secrets detected. Data appears clean.")
    else:
        total_secrets = sum(stats.values())

        for secret_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_secrets * 100) if total_secrets > 0 else 0
            print(f"  {secret_type:20s}: {count:6d} occurrences ({pct:5.1f}%)")

            files = affected_files[secret_type]
            show = files[:3]
            for f in show:
                print(f"    - {f}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more files")
            print()

        print(f"{'TOTAL':22s}: {total_secrets:6d}")

    print("=" * 60)

    if dry_run and stats:
        print("\nTo sanitize the data, run without --dry-run flag")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sanitize IaC training data by removing secrets and sensitive information"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.expanduser("~/.cache/nanochat/base_data"),
        help="Input directory with parquet shards (default: ~/.cache/nanochat/base_data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for sanitized shards (default: same as input-dir, in-place)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Also scan and sanitize raw IaC files (.tf, .yaml, .yml) in this directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report findings without modifying files",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    raw_dir = Path(args.raw_dir) if args.raw_dir else None

    print("=" * 60)
    print("IaC Data Sanitization Tool")
    print("=" * 60)
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if raw_dir:
        print(f"Raw files directory: {raw_dir}")
    print()

    # Sanitize parquet shards
    stats, affected_files = sanitize_parquet_shards(input_dir, output_dir, args.dry_run)

    # Sanitize raw files if requested
    if raw_dir:
        raw_stats, raw_affected = sanitize_raw_files(raw_dir, args.dry_run)
        for secret_type, count in raw_stats.items():
            stats[secret_type] += count
        for secret_type, files in raw_affected.items():
            affected_files[secret_type].extend(files)

    # Print final report
    print_report(stats, affected_files, args.dry_run)


if __name__ == "__main__":
    main()
