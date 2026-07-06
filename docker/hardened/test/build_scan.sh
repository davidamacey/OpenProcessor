#!/usr/bin/env bash
# Build an image and scan it for CRITICAL/HIGH CVEs with Trivy.
# usage: build_scan.sh <dockerfile> <context> <tag> [trivyignore]
set -euo pipefail

DF="${1:?dockerfile}"; CTX="${2:?context}"; TAG="${3:?tag}"; IGN="${4:-}"
OUT="/tmp/trisec/$(echo "$TAG" | tr '/:' '__')"

echo "==> docker build $TAG"
docker build -f "$DF" -t "$TAG" "$CTX"

echo "==> trivy scan $TAG (CRITICAL,HIGH)"
IGN_ARG=(); [ -n "$IGN" ] && IGN_ARG=(--ignorefile "$IGN")
trivy image --scanners vuln --severity CRITICAL,HIGH --skip-version-check \
    "${IGN_ARG[@]}" --format json -o "${OUT}.json" "$TAG"

python3 - "$OUT.json" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1]))
sev = collections.Counter(); rows = collections.defaultdict(set)
for r in d.get("Results", []):
    for v in r.get("Vulnerabilities", []) or []:
        sev[v["Severity"]] += 1
        rows[v["Severity"]].add((v["VulnerabilityID"], v.get("PkgName")))
print(f"CRITICAL={sev['CRITICAL']}  HIGH={sev['HIGH']}")
for s in ("CRITICAL", "HIGH"):
    pkgs = collections.Counter(p for _, p in rows[s])
    if pkgs:
        print(f"  {s} by package:", dict(pkgs.most_common()))
PY
echo "==> report: ${OUT}.json"
