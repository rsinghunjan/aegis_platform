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
# Terraform skeleton (cloud-agnostic) — quickstart

This folder contains a minimal Terraform skeleton showing how to structure provider-agnostic modules
and per-provider overlays. It's a starting point: fill in provider credentials & specifics with your infra team.

Structure
- modules/
  - object_store/         # exposes bucket name, endpoint
  - database/             # exposes database URL
  - k8s_cluster/          # creates a k8s cluster (provider-specific)
- overlays/
  - aws/
  - gcp/
  - azure/

Quick usage (dev)
1. Install terraform (>=1.4)
2. cd infra/terraform/overlays/aws
3. terraform init
4. terraform plan

The modules intentionally leave provider configuration to the overlay (so you can swap cloud providers).
See the sample modules for required variables and outputs.
infra/terraform/README.md
