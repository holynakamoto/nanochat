---
description: Audit project dependencies for known vulnerabilities
---

To audit dependencies for vulnerabilities in this project:

1. **Identify Dependencies**: Check `pyproject.toml` and `uv.lock` to understand the current dependency tree.
2. **Install Audit Tool**: If not already present, use `pip-audit`. You can run it via uv:
   ```bash
   uv run pip install pip-audit
   ```
3. **Run Audit**:
   - For a quick check of the local environment:
     ```bash
     uv run pip-audit --local
     ```
   - For a more thorough check against the locked dependencies (most accurate):
     ```bash
     uv export --format requirements.txt > temp_reqs.txt && uv run pip-audit -r temp_reqs.txt; rm temp_reqs.txt
     ```
4. **Analysis**:
   - Review any reported CVEs.
   - For each vulnerability, check if a newer version of the package is available that fixes it.
   - Propose updates to `pyproject.toml` if necessary.
5. **Remediation**:
   - Update vulnerable packages using `uv add package@latest` or manual edits followed by `uv lock`.
   - Run tests (`pytest`) to ensure no regressions were introduced by the updates.
