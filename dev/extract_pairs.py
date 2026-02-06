"""
Extract Requirement->Code pairs from scraped IaC repositories.

This script extracts training pairs that teach IaC-GPT to map natural language
intent to Infrastructure-as-Code. It processes Terraform, Kubernetes, Ansible,
Crossplane, and Docker files to create requirement->code mappings.

Extraction Methods:
1. Variable Descriptions -> Code Context (Terraform)
2. Resource Block -> Inferred Requirement (Terraform)
3. Kubernetes Manifest -> Requirement (K8s YAML)
4. Ansible Task -> Requirement (Ansible YAML)
5. Multi-resource Module -> Composite Requirement (Terraform)

Usage:
    python dev/extract_pairs.py --input-dir data/iac_raw_cloned --output data/iac_pairs.jsonl
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import yaml


# Resource type to human-readable description mapping
RESOURCE_TYPE_MAP = {
    # AWS Resources
    "aws_vpc": "AWS VPC (Virtual Private Cloud)",
    "aws_subnet": "AWS subnet",
    "aws_security_group": "AWS security group",
    "aws_iam_role": "AWS IAM role",
    "aws_iam_policy": "AWS IAM policy",
    "aws_s3_bucket": "AWS S3 bucket",
    "aws_eks_cluster": "AWS EKS Kubernetes cluster",
    "aws_rds_instance": "AWS RDS database instance",
    "aws_lambda_function": "AWS Lambda function",
    "aws_instance": "AWS EC2 instance",
    "aws_route_table": "AWS route table",
    "aws_internet_gateway": "AWS internet gateway",
    "aws_nat_gateway": "AWS NAT gateway",
    "aws_eip": "AWS Elastic IP",
    "aws_lb": "AWS load balancer",
    "aws_autoscaling_group": "AWS autoscaling group",
    "aws_launch_template": "AWS launch template",

    # GCP Resources
    "google_compute_instance": "GCP compute instance",
    "google_container_cluster": "GCP GKE Kubernetes cluster",
    "google_compute_network": "GCP VPC network",
    "google_compute_subnetwork": "GCP subnetwork",
    "google_compute_firewall": "GCP firewall rule",
    "google_storage_bucket": "GCP Cloud Storage bucket",
    "google_sql_database_instance": "GCP Cloud SQL database instance",

    # Azure Resources
    "azurerm_virtual_network": "Azure virtual network",
    "azurerm_subnet": "Azure subnet",
    "azurerm_kubernetes_cluster": "Azure AKS Kubernetes cluster",
    "azurerm_resource_group": "Azure resource group",
    "azurerm_storage_account": "Azure storage account",
    "azurerm_virtual_machine": "Azure virtual machine",
}


def parse_resource_type(resource_type: str) -> str:
    """
    Convert a resource type to human-readable description.
    Falls back to parsing the name for unknown types.
    """
    if resource_type in RESOURCE_TYPE_MAP:
        return RESOURCE_TYPE_MAP[resource_type]

    # Parse unknown types: aws_xxx_yyy -> AWS xxx yyy
    parts = resource_type.split("_")
    if len(parts) >= 2:
        provider = parts[0].upper()
        resource_parts = parts[1:]
        resource_name = " ".join(resource_parts)
        return f"{provider} {resource_name}"

    return resource_type


def extract_hcl_blocks(content: str, block_type: str) -> List[Dict[str, str]]:
    """
    Extract HCL blocks from Terraform content using regex and brace matching.
    Returns list of dicts with 'type', 'name', and 'body'.
    """
    blocks = []

    # Pattern matches: block_type "type_value" "name_value" {
    # or: block_type "name_value" {
    if block_type in ["variable", "output", "locals"]:
        pattern = rf'{block_type}\s+"([^"]+)"\s*{{'
    else:
        pattern = rf'{block_type}\s+"([^"]+)"\s+"([^"]+)"\s*{{'

    for match in re.finditer(pattern, content):
        start_pos = match.end() - 1  # Position of opening brace

        # Find matching closing brace
        brace_count = 1
        pos = start_pos + 1
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            body = content[start_pos:pos]

            if block_type in ["variable", "output", "locals"]:
                blocks.append({
                    "type": block_type,
                    "name": match.group(1),
                    "body": body,
                    "full_block": match.group(0)[:-1] + body
                })
            else:
                blocks.append({
                    "type": match.group(1),
                    "name": match.group(2),
                    "body": body,
                    "full_block": match.group(0)[:-1] + body
                })

    return blocks


def extract_variable_description_pairs(content: str, source_file: str) -> List[Dict]:
    """Extraction Method 1: Variable Descriptions -> Code Context"""
    pairs = []
    variables = extract_hcl_blocks(content, "variable")

    for var in variables:
        desc_match = re.search(r'description\s*=\s*"([^"]*)"', var["body"])
        if desc_match:
            description = desc_match.group(1)

            type_match = re.search(r'type\s*=\s*(\w+)', var["body"])
            default_match = re.search(r'default\s*=\s*"([^"]*)"', var["body"])

            requirement = f"Define a variable for {description}"
            if type_match:
                requirement += f" with type {type_match.group(1)}"
            if default_match:
                requirement += f" and default {default_match.group(1)}"

            pairs.append({
                "requirement": requirement,
                "code": var["full_block"],
                "provider": "terraform",
                "pair_type": "variable_description",
                "source_file": source_file
            })

    return pairs


def extract_resource_inferred_pairs(content: str, source_file: str) -> List[Dict]:
    """Extraction Method 2: Resource Block -> Inferred Requirement"""
    pairs = []
    resources = extract_hcl_blocks(content, "resource")

    for resource in resources:
        resource_type = resource["type"]
        resource_name = resource["name"]
        resource_desc = parse_resource_type(resource_type)

        requirement = f"Create a {resource_desc} resource named '{resource_name}'"

        body = resource["body"]

        if "cidr_block" in body:
            requirement += " with configurable CIDR block"

        if "enable_dns_hostnames" in body and "= true" in body:
            requirement += " and DNS hostnames enabled"

        if "enable_dns_support" in body and "= true" in body:
            requirement += " and DNS support enabled"

        if "replicas" in body:
            replicas_match = re.search(r'replicas\s*=\s*(\d+)', body)
            if replicas_match:
                requirement += f" with {replicas_match.group(1)} replicas"

        pairs.append({
            "requirement": requirement,
            "code": resource["full_block"],
            "provider": "terraform",
            "pair_type": "resource_inferred",
            "source_file": source_file
        })

    return pairs


def extract_module_composite_pairs(content: str, source_file: str) -> List[Dict]:
    """Extraction Method 5: Multi-resource Module -> Composite Requirement"""
    pairs = []
    resources = extract_hcl_blocks(content, "resource")

    if len(resources) < 2:
        return pairs

    resource_types = [r["type"] for r in resources]

    vpc_resources = ["aws_vpc", "aws_subnet", "aws_internet_gateway", "aws_route_table"]
    eks_resources = ["aws_eks_cluster", "aws_iam_role", "aws_security_group"]

    has_vpc_module = any(rt in resource_types for rt in vpc_resources)
    has_eks_module = any(rt in resource_types for rt in eks_resources)

    if has_vpc_module and len([rt for rt in resource_types if rt in vpc_resources]) >= 2:
        components = []
        if "aws_vpc" in resource_types:
            components.append("VPC")
        if "aws_subnet" in resource_types:
            components.append("subnets")
        if "aws_internet_gateway" in resource_types:
            components.append("internet gateway")
        if "aws_route_table" in resource_types:
            components.append("route tables")

        requirement = "Create a VPC module with " + ", ".join(components)

        pairs.append({
            "requirement": requirement,
            "code": content,
            "provider": "terraform",
            "pair_type": "module_composite",
            "source_file": source_file
        })

    elif has_eks_module and len([rt for rt in resource_types if rt in eks_resources]) >= 2:
        pairs.append({
            "requirement": "Create an EKS cluster module with IAM roles and security groups",
            "code": content,
            "provider": "terraform",
            "pair_type": "module_composite",
            "source_file": source_file
        })

    return pairs


def extract_kubernetes_pairs(content: str, source_file: str) -> List[Dict]:
    """Extraction Method 3: Kubernetes Manifest -> Requirement"""
    pairs = []

    try:
        documents = list(yaml.safe_load_all(content))

        for doc in documents:
            if not doc or not isinstance(doc, dict):
                continue

            kind = doc.get("kind", "")
            metadata = doc.get("metadata", {})
            spec = doc.get("spec", {})

            if not kind:
                continue

            name = metadata.get("name", "unnamed") if isinstance(metadata, dict) else "unnamed"
            requirement = f"Create a Kubernetes {kind}"

            if name != "unnamed":
                requirement += f" named '{name}'"

            if isinstance(spec, dict):
                if kind == "Deployment" and "replicas" in spec:
                    requirement += f" with {spec['replicas']} replicas"

                if kind == "Service":
                    service_type = spec.get("type", "ClusterIP")
                    requirement += f" of type {service_type}"

            if kind == "Namespace":
                requirement = f"Create a Kubernetes namespace named '{name}'"
            elif kind == "ConfigMap":
                requirement = f"Create a Kubernetes ConfigMap named '{name}'"
            elif kind == "Secret":
                requirement = f"Create a Kubernetes Secret named '{name}'"

            pairs.append({
                "requirement": requirement,
                "code": yaml.dump(doc, default_flow_style=False, sort_keys=False),
                "provider": "kubernetes",
                "pair_type": "manifest_inferred",
                "source_file": source_file
            })

    except yaml.YAMLError:
        pass

    return pairs


def extract_ansible_pairs(content: str, source_file: str) -> List[Dict]:
    """Extraction Method 4: Ansible Task -> Requirement"""
    pairs = []

    try:
        documents = list(yaml.safe_load_all(content))

        for doc in documents:
            if not doc:
                continue

            if isinstance(doc, dict) and "tasks" in doc:
                tasks = doc["tasks"]
            elif isinstance(doc, list):
                tasks = doc
            else:
                continue

            if not isinstance(tasks, list):
                continue

            for task in tasks:
                if not isinstance(task, dict):
                    continue

                task_name = task.get("name", "")
                if not task_name:
                    continue

                requirement = task_name

                task_copy = dict(task)
                task_copy.pop("name", None)

                code = f"- name: {task_name}\n"
                code += yaml.dump(task_copy, default_flow_style=False, indent=2)

                pairs.append({
                    "requirement": requirement,
                    "code": code,
                    "provider": "ansible",
                    "pair_type": "task_name",
                    "source_file": source_file
                })

    except yaml.YAMLError:
        pass

    return pairs


def extract_crossplane_pairs(content: str, source_file: str) -> List[Dict]:
    """Extract pairs from Crossplane YAML manifests."""
    pairs = []

    try:
        documents = list(yaml.safe_load_all(content))

        for doc in documents:
            if not doc or not isinstance(doc, dict):
                continue

            kind = doc.get("kind", "")
            metadata = doc.get("metadata", {})

            if not kind:
                continue

            name = metadata.get("name", "unnamed") if isinstance(metadata, dict) else "unnamed"
            requirement = f"Create a Crossplane {kind}"

            if name != "unnamed":
                requirement += f" named '{name}'"

            pairs.append({
                "requirement": requirement,
                "code": yaml.dump(doc, default_flow_style=False, sort_keys=False),
                "provider": "crossplane",
                "pair_type": "manifest_inferred",
                "source_file": source_file
            })

    except yaml.YAMLError:
        pass

    return pairs


def extract_docker_pairs(content: str, source_file: str) -> List[Dict]:
    """Extract pairs from Dockerfiles."""
    pairs = []

    lines = content.split("\n")

    base_image = None
    has_run = False
    has_copy = False
    has_env = False

    for line in lines:
        line = line.strip()
        if line.startswith("FROM "):
            base_image = line.split()[1] if len(line.split()) > 1 else None
        elif line.startswith("RUN "):
            has_run = True
        elif line.startswith("COPY "):
            has_copy = True
        elif line.startswith("ENV "):
            has_env = True

    if base_image:
        requirement = f"Create a Dockerfile based on {base_image}"

        details = []
        if has_run:
            details.append("with custom commands")
        if has_copy:
            details.append("copying application files")
        if has_env:
            details.append("setting environment variables")

        if details:
            requirement += " " + ", ".join(details)

        pairs.append({
            "requirement": requirement,
            "code": content,
            "provider": "docker",
            "pair_type": "dockerfile_inferred",
            "source_file": source_file
        })

    return pairs


def process_file(file_path: Path, category: str) -> List[Dict]:
    """Process a single file based on its category."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []

    source_file = file_path.name

    if category == "terraform":
        pairs = []
        pairs.extend(extract_variable_description_pairs(content, source_file))
        pairs.extend(extract_resource_inferred_pairs(content, source_file))
        pairs.extend(extract_module_composite_pairs(content, source_file))
        return pairs
    elif category == "kubernetes":
        return extract_kubernetes_pairs(content, source_file)
    elif category == "ansible":
        return extract_ansible_pairs(content, source_file)
    elif category == "crossplane":
        return extract_crossplane_pairs(content, source_file)
    elif category == "docker":
        return extract_docker_pairs(content, source_file)

    return []


def filter_pairs(pairs: List[Dict], min_code_length: int, max_code_length: int) -> List[Dict]:
    """Filter pairs based on code length constraints."""
    filtered = []

    for pair in pairs:
        code_length = len(pair["code"])

        if code_length < min_code_length:
            continue

        if code_length > max_code_length:
            pair["code"] = pair["code"][:max_code_length]

        filtered.append(pair)

    return filtered


def print_statistics(stats: Dict, total_pairs: int):
    """Print extraction statistics."""
    print("\n" + "=" * 70)
    print("Extraction Statistics")
    print("=" * 70)

    print("\nPairs by Provider:")
    print("-" * 70)
    for provider in ["terraform", "kubernetes", "ansible", "crossplane", "docker"]:
        count = stats["by_provider"].get(provider, 0)
        percentage = (count / total_pairs * 100) if total_pairs > 0 else 0
        print(f"{provider:15s}: {count:6d} pairs ({percentage:5.1f}%)")

    print("\nPairs by Type:")
    print("-" * 70)
    for pair_type, count in sorted(stats["by_type"].items()):
        percentage = (count / total_pairs * 100) if total_pairs > 0 else 0
        print(f"{pair_type:25s}: {count:6d} pairs ({percentage:5.1f}%)")

    print("\n" + "=" * 70)
    print(f"{'TOTAL PAIRS':25s}: {total_pairs:6d}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Extract Requirement->Code pairs from IaC repositories")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/iac_raw_cloned",
        help="Input directory with scraped IaC files organized by type (default: data/iac_raw_cloned)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/iac_pairs.jsonl",
        help="Output JSONL file path (default: data/iac_pairs.jsonl)",
    )
    parser.add_argument(
        "--min-code-length",
        type=int,
        default=50,
        help="Skip pairs where code < N characters (default: 50)",
    )
    parser.add_argument(
        "--max-code-length",
        type=int,
        default=5000,
        help="Truncate code > N characters (default: 5000)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed statistics at end",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    print("=" * 70)
    print("IaC Requirement->Code Pair Extraction")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output file:      {output_path}")
    print(f"Min code length:  {args.min_code_length}")
    print(f"Max code length:  {args.max_code_length}")
    print("=" * 70)

    if not input_dir.exists():
        print(f"\nERROR: Input directory {input_dir} does not exist!")
        print("Please run the IaC scraping script first.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    stats = {
        "by_provider": defaultdict(int),
        "by_type": defaultdict(int),
    }

    # Process each IaC category
    categories = [
        ("terraform", "*.tf"),
        ("kubernetes", "*.yaml"),
        ("ansible", "*.yaml"),
        ("crossplane", "*.yaml"),
        ("docker", "Dockerfile*"),
    ]

    for category_name, file_pattern in categories:
        category_dir = input_dir / category_name

        if not category_dir.exists():
            print(f"\nWarning: {category_dir} does not exist, skipping...")
            continue

        print(f"\nProcessing {category_name} files...")

        file_count = 0
        pair_count = 0

        for file_path in category_dir.glob(file_pattern):
            if not file_path.is_file():
                continue

            pairs = process_file(file_path, category_name)
            all_pairs.extend(pairs)

            file_count += 1
            pair_count += len(pairs)

        print(f"  Processed {file_count:4d} files -> {pair_count:5d} pairs")

    # Filter pairs by code length
    print(f"\nFiltering pairs (min={args.min_code_length}, max={args.max_code_length})...")
    filtered_pairs = filter_pairs(all_pairs, args.min_code_length, args.max_code_length)

    print(f"  Before filtering: {len(all_pairs):5d} pairs")
    print(f"  After filtering:  {len(filtered_pairs):5d} pairs")

    # Calculate statistics
    for pair in filtered_pairs:
        stats["by_provider"][pair["provider"]] += 1
        stats["by_type"][pair["pair_type"]] += 1

    # Write output JSONL
    print(f"\nWriting pairs to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in filtered_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"  Wrote {len(filtered_pairs)} pairs")

    # Print statistics
    print_statistics(stats, len(filtered_pairs))

    # Calculate output size
    if output_path.exists() and output_path.stat().st_size > 0:
        output_size_mb = output_path.stat().st_size / 1_000_000
        print(f"\nOutput file size: {output_size_mb:.2f} MB")

    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
