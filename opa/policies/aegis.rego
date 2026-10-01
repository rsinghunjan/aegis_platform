package aegis

# Pattern 2: return a single object at data.aegis.decision
default decision = {
  "allow": false,
  "reason": "default_deny",
  "policy_id": "aegis-default",
  "policy_version": "0.1.0",
  "obligations": [],
  "labels": {}
}

is_high_risk {
  input.environment.risk_tier == "prod"
} else {
  input.environment.risk_tier == "hipaa"
} else {
  input.environment.risk_tier == "pci"
}

decision = d {
  allow := true
  d := {
    "allow": allow,
    "reason": "allowed_by_minimal_policy",
    "policy_id": "aegis-minimal",
    "policy_version": "0.1.0",
    "obligations": obligations,
    "labels": {
      "risk_tier": input.environment.risk_tier,
      "data_classification": input.resource.attributes.data_classification
    }
  }

  obligations := []

  # Example: require approvals for deploy.request in high-risk envs.
  input.request.action == "deploy.request"
  is_high_risk

  obligations := [
    {"type": "require_approvals", "count": 2, "scope": "environment"},
    {"type": "record_change_ticket"},
  ]
}

decision = d {
  # Allow job.submit in non-high-risk without extra obligations.
  input.request.action == "job.submit"
  not is_high_risk
  d := {
    "allow": true,
    "reason": "job_submit_allowed_low_risk",
    "policy_id": "aegis-minimal",
    "policy_version": "0.1.0",
    "obligations": [],
    "labels": {
      "risk_tier": input.environment.risk_tier,
      "data_classification": input.resource.attributes.data_classification
    }
  }
}

decision = d {
  # Deny PHI/PCI jobs unless env tier matches.
  input.request.action == "job.submit"
  cls := input.resource.attributes.data_classification
  (cls == "phi"  ; input.environment.risk_tier != "hipaa") ||
  (cls == "pci"  ; input.environment.risk_tier != "pci")

  d := {
    "allow": false,
    "reason": "classification_requires_matching_env_tier",
    "policy_id": "aegis-minimal",
    "policy_version": "0.1.0",
    "obligations": [
      {"type": "restrict_environment", "required_tier": cls}
    ],
    "labels": {
      "risk_tier": input.environment.risk_tier,
      "data_classification": cls
    }
  }
}
