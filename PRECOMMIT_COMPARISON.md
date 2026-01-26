# Pre-commit Configuration Comparison

**Comparison between:** triton-api vs transcribe-app (OpenTranscribe)

## Summary

✅ **Triton API has a MORE comprehensive pre-commit configuration than transcribe-app**

The triton-api configuration includes all the same security and quality checks as OpenTranscribe, plus additional checks and newer tool versions.

---

## Tool Versions Comparison

| Tool | Triton API | Transcribe-App | Winner |
|------|-----------|----------------|---------|
| pre-commit-hooks | v5.0.0 | v4.5.0 | ✅ Triton (newer) |
| ruff | v0.14.0 | v0.2.2 | ✅ Triton (much newer) |
| mypy | v1.13.0 | v1.8.0 | ✅ Triton (newer) |
| hadolint | v2.12.0 | v2.12.0 | ✅ Same |
| gitleaks | v8.21.2 | v8.18.1 | ✅ Triton (newer) |
| bandit | 1.8.0 | 1.7.7 | ✅ Triton (newer) |
| shellcheck-py | v0.10.0.1 | v0.9.0.6 | ✅ Triton (newer) |
| conventional-pre-commit | v3.6.0 | v3.0.0 | ✅ Triton (much newer) |

---

## Hooks Comparison

### ✅ Hooks in BOTH configs

1. **File Quality Checks** (pre-commit-hooks):
   - trailing-whitespace
   - end-of-file-fixer
   - check-yaml (with --unsafe for docker-compose)
   - check-json
   - check-toml
   - check-added-large-files (10MB limit in both)
   - check-merge-conflict
   - check-case-conflict (triton only, but good practice)
   - mixed-line-ending (--fix=lf)
   - check-executables-have-shebangs
   - check-shebang-scripts-are-executable

2. **Python Quality** (ruff):
   - ruff linter (--fix, --show-fixes)
   - ruff-format

3. **Python Type Checking** (mypy):
   - Type checking with ignore-missing-imports
   - check-untyped-defs

4. **Dockerfile Linting** (hadolint):
   - Lint Dockerfiles with system hadolint

5. **Security** (gitleaks):
   - Secret detection

6. **Python Security** (bandit):
   - Security vulnerability scanning
   - Configured via pyproject.toml

7. **Shell Scripts** (shellcheck):
   - Bash script linting (--severity=warning)

8. **Commit Messages** (conventional-pre-commit):
   - Conventional commit message validation

---

## ✅ Additional Hooks in Triton API (NOT in transcribe-app)

1. **detect-private-key**: Extra security check for accidentally committed keys
2. **pygrep-hooks**: Additional Python-specific checks:
   - python-check-blanket-noqa
   - python-check-blanket-type-ignore
   - python-no-eval
   - python-use-type-annotations

---

## ⚠️ Configuration Differences

### Language Target
- **Triton API**: `python: python3.12` (explicitly set)
- **Transcribe-app**: Not specified (defaults to system Python)

### Excludes
- **Triton API**: More comprehensive with regex pattern excluding:
  - All standard Python cache dirs
  - `reference_repos/`, `pytorch_models/`, `models/`, `cache/`
  - `benchmarks/results/`, `outputs/`, `test_images/`
  - `security-reports/`

- **Transcribe-app**: Only excludes `security-reports/` in individual hooks

### Conventional Commits
- **Both**: Now have `--force-scope` (just added to triton-api)

---

## Tool-Specific Configuration

### MyPy

**Triton API** (pyproject.toml):
```toml
[tool.mypy]
python_version = "3.12"
explicit_package_bases = true
mypy_path = "."
namespace_packages = true
# ... extensive third-party ignores
```

**Transcribe-app**: Basic configuration in pre-commit hook only

### Bandit

**Both** use pyproject.toml configuration:

**Triton API**:
```toml
[tool.bandit]
exclude_dirs = ["tests", ".venv", "venv", "reference_repos", "pytorch_models", "cache"]
skips = [
    "B101",  # assert_used - common in test files
    "B311",  # random - only used for non-cryptographic retry jitter
    "B601",  # paramiko_calls - not used
]
```

**Transcribe-app**:
```toml
[tool.bandit]
exclude_dirs = ["tests", "venv", ".venv", "node_modules"]
skips = ["B101", "B601"]
```

**Winner**: Triton API (more comprehensive excludes, documents B311 skip)

### Ruff

**Triton API**: Extensive configuration in pyproject.toml with:
- 40+ enabled rule categories
- Per-file ignores for different code types
- Detailed documentation for each ignored rule

**Transcribe-app**: Simpler configuration

---

## Coverage Comparison

### File Types Checked

| Type | Triton API | Transcribe-App |
|------|-----------|----------------|
| Python | ✅ Yes | ✅ Yes |
| Shell Scripts | ✅ Yes | ✅ Yes |
| Dockerfiles | ✅ Yes | ✅ Yes |
| YAML/JSON/TOML | ✅ Yes | ✅ Yes |
| JavaScript/Frontend | ❌ No (not needed) | ✅ Yes (has frontend) |

---

## Recommendation

**No changes needed to triton-api** - The configuration is already more comprehensive than transcribe-app.

The only item added was `--force-scope` to conventional-pre-commit to match OpenTranscribe's strictness.

### Why Triton API Config is Better

1. **Newer tool versions** - Gets latest bug fixes and features
2. **Additional security** - Extra private key detection
3. **More Python checks** - pygrep-hooks catch more issues
4. **Better excludes** - Comprehensive regex pattern prevents false positives
5. **Explicit Python version** - Ensures consistency across environments
6. **Better documentation** - All configs have comments explaining why

---

## Current Status

✅ All hooks configured and working
✅ Bandit security scanning enabled
✅ All tool versions up-to-date
✅ Comprehensive exclude patterns
✅ Conventional commits with scope enforcement
✅ Type checking with mypy
✅ Code formatting with ruff
✅ Security scanning with gitleaks + bandit
✅ Shell script linting with shellcheck
✅ Dockerfile linting with hadolint

**Result**: Professional-grade pre-commit configuration exceeding OpenTranscribe standards.
