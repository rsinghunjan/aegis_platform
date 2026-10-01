  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
```markdown
# Gatekeeper (OPA) — Aegis Signed Artifact Constraint (audit mode)

Purpose
- Deploy this ConstraintTemplate + Constraint to detect Pods that would violate your "only run signed artifacts" policy without blocking them (dry‑run).
- Use audit mode to discover impacted Deployments/DaemonSets/StatefulSets and update their PodTemplates before enabling enforcement.

Files
- constrainttemplate_aegis_signed.yaml — ConstraintTemplate with Rego that checks Pod objects.
- constraint_aegis_signed_audit.yaml — Constraint instance (enforcementAction: dryrun); edit namespaces to protect.
- examples/pod_unsigned_example.yaml — sample Pod that will produce a violation (no verify initContainer and no cosign-public-key).
- examples/pod_signed_example.yaml — sample Pod that satisfies the policy.
- README.md — this file.

Prereqs
- Gatekeeper v3+ installed in the cluster:
  kubectl apply -f https://raw.githubusercontent.com/open-policy-agent/gatekeeper/master/deploy/gatekeeper.yaml

Install (staging / audit)
1) Apply ConstraintTemplate:
   kubectl apply -f k8s/gatekeeper/constrainttemplate_aegis_signed.yaml

2) Apply Constraint in audit mode:
   kubectl apply -f k8s/gatekeeper/constraint_aegis_signed_audit.yaml

3) Verify Constraint was created:
   kubectl get aegissignedpod.require-aegis-signed-audit -o yaml

Testing (discover violations in dry-run)
- Apply example unsigned Pod (will be admitted but cause an audit violation record):
   kubectl apply -f k8s/gatekeeper/examples/pod_unsigned_example.yaml

- List Gatekeeper violations:
   kubectl get violations --all-namespaces
   # Or:
   kubectl get aegissignedpod.require-aegis-signed-audit -o yaml

- Describe the constraint to see status and violations:
   kubectl describe constraint aegissignedpod.require-aegis-signed-audit

- Apply a compliant Pod (signed example) — it should be admitted and not produce a violation:
   kubectl apply -f k8s/gatekeeper/examples/pod_signed_example.yaml

Iterate
- Review violations and update the offending Deployment/PodTemplate to include the required initContainer and secret mount.
- Repeat until violations are resolved for a namespace.

Flip to enforce (only after audit period & fixes)
- Change enforcementAction from dryrun to deny in the Constraint and apply:
   # edit k8s/gatekeeper/constraint_aegis_signed_audit.yaml -> enforcementAction: deny
   kubectl apply -f k8s/gatekeeper/constraint_aegis_signed_audit.yaml

Notes & tips
- Scope the Constraint narrowly when enabling enforcement (start with one namespace).
- Consider an "allowlist" annotation workflow for privileged namespaces (kube-system, ingress controllers).
- If you use other secret names or different initContainer naming conventions, update the Rego in the ConstraintTemplate accordingly.
- Gatekeeper reports can be noisy initially; use the audit period to contact owners of non-compliant workloads and create migration tasks.

If you want, I can:
- A) open a PR adding these files to your repo on branch feature/gatekeeper-audit and include a short remediation checklist and suggested owners,
- B) add a Helm values patch to deploy the Constraint + audit-only setting via Helm, or
- C) extend the Rego to also check for Rekor entry annotations (require an annotation rekor-entry: "<id>") and report if missing.

Pick A, B, or C (or ask another next step) and I’ll prepare it.
k8s/gatekeeper/README.md
