"""Tests for services/cloud: cost tracking, secrets, deployment, autoscaling."""
import pytest

from services.cloud import (
    CostTracker,
    Budget,
    BudgetExceededError,
    EnvSecretManager,
    ModelDeployment,
    DeploymentSpec,
    DeploymentTarget,
    AutoScalingPolicy,
    ScalingDecision,
)


def test_cost_tracker_records_and_sums_spend():
    tracker = CostTracker()
    tracker.record("tenant-a", 1.5, "inference call 1")
    total = tracker.record("tenant-a", 2.5, "inference call 2")
    assert total == 4.0
    assert tracker.spend("tenant-a") == 4.0
    assert len(tracker.entries("tenant-a")) == 2


def test_cost_tracker_enforces_budget():
    tracker = CostTracker()
    tracker.set_budget(Budget(scope="tenant-a", limit=5.0))
    tracker.record("tenant-a", 4.0)
    assert tracker.remaining_budget("tenant-a") == pytest.approx(1.0)
    with pytest.raises(BudgetExceededError):
        tracker.record("tenant-a", 2.0)


def test_env_secret_manager_reads_prefixed_env_var(monkeypatch):
    monkeypatch.setenv("AEGIS_SECRET_API_KEY", "configured-test-value")
    manager = EnvSecretManager()
    assert manager.get_secret("api_key") == "configured-test-value"
    with pytest.raises(KeyError):
        manager.get_secret("missing_key")
    assert manager.get_secret_or_default("missing_key", "fallback") == "fallback"


def test_model_deployment_lifecycle():
    manager = ModelDeployment()
    spec = DeploymentSpec(model_name="demo", model_version="1", target=DeploymentTarget.LOCAL)
    deployment = manager.deploy(spec)
    assert deployment.status.value == "running"
    assert deployment.endpoint == "local://demo:1"

    fetched = manager.get(deployment.deployment_id)
    assert fetched is deployment

    terminated = manager.terminate(deployment.deployment_id)
    assert terminated.status.value == "terminated"


def test_model_deployment_handles_deploy_failure():
    manager = ModelDeployment()
    spec = DeploymentSpec(model_name="demo", model_version="1", target=DeploymentTarget.EC2)

    def failing_deploy(_spec):
        raise RuntimeError("provisioning failed")

    with pytest.raises(RuntimeError):
        manager.deploy(spec, deploy_fn=failing_deploy)


def test_autoscaling_policy_decisions():
    policy = AutoScalingPolicy(min_replicas=1, max_replicas=5, target_utilization=0.7, tolerance=0.1)
    assert policy.decide(2, 0.95) == ScalingDecision.SCALE_UP
    assert policy.decide(2, 0.2) == ScalingDecision.SCALE_DOWN
    assert policy.decide(2, 0.7) == ScalingDecision.HOLD

    assert policy.next_replica_count(2, 0.95) >= 2
    assert policy.next_replica_count(1, 0.2) >= policy.min_replicas
    assert policy.next_replica_count(10, 0.95) <= policy.max_replicas
